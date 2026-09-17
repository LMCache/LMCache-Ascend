// SPDX-License-Identifier: Apache-2.0

// C++ host layer for the block-level MP-mode KV transfer (mirrors upstream
// LMCache csrc/mp_mem_kernels.cu).
//
// Design contract:
//   - prepare_group resolves a PageBufferShapeDesc + EngineKVFormat into the
//     flat BlockTransferLayout POD ONCE per group, with every static geometry
//     check. Both native entry points consume only PreparedGroup; the
//     device kernel receives the POD verbatim and never re-interprets the
//     format.
//   - validate_launch checks the per-transfer dynamic variables.
//   - The plan executor validates ALL groups and ALL launches before any
//     staging copy or kernel launch is enqueued: an unsupported group
//     fails with "nothing started", never half-enqueued.
//   - The kernel is a pure byte mover; there is no packed/two-plane special
//     case and no dtype dispatch left in this file.

#include "mp_mem_kernels.h"

#include "kernels/multi_layer/multi_layer_block_mem_kernels.h"
#include "utils.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// DataCopy granularity on the GM side: segment base addresses and segment
// byte lengths must both be multiples of 32B.
constexpr int64_t kGmAlignBytes = 32;
// Single UB segment budget: the depth-2 queue must hold two segments, so one
// segment may use at most half of the total budget.
constexpr int64_t kUbSegmentBytes = kvcache_ops::kBlockTransferUbBytes /
                                    kvcache_ops::kBlockTransferQueueDepth;
// DataCopyPad GM-side gap fields are uint32.
constexpr int64_t kCopyGmGapMax =
    static_cast<int64_t>(std::numeric_limits<uint32_t>::max());

int64_t align_up32(int64_t v) { return (v + 31) & ~int64_t(31); }

int64_t checked_mul(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0, what, " must be non-negative, got ", a, " * ",
              b);
  if (a != 0) {
    TORCH_CHECK(b <= std::numeric_limits<int64_t>::max() / a, what,
                " overflows int64: ", a, " * ", b);
  }
  return a * b;
}

int64_t checked_add(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0 && b <= std::numeric_limits<int64_t>::max() - a,
              what, " overflows int64: ", a, " + ", b);
  return a + b;
}

// ---------------------------------------------------------------------------
// prepare_group: static geometry, resolved once per group.
// ---------------------------------------------------------------------------

struct PreparedGroup {
  kvcache_ops::BlockTransferLayout layout{};
  bool separate_plane = false;  // format 16: (layer, plane, block) work items
  int32_t nl = 0;
  int32_t nb = 0;
  int32_t bs = 0;
  int64_t slots_per_object = 0;
};

// num_planes == 0 is the "unfilled" sentinel: only legacy 16/13 inputs may be
// derived (dense token rows, per-plane byte geometry from the scalar fields).
// Format 17 must carry explicit validated geometry from registration.
PageBufferShapeDesc describe_legacy_dense_token_rows(PageBufferShapeDesc sd,
                                                     EngineKVFormat fmt) {
  const int64_t row = checked_mul(
      checked_mul(sd.nh, sd.hs, "nh * hs"), sd.element_size,
      "legacy scalar row bytes");
  // Legacy block_stride_elems is in ELEMENTS of the original dtype.
  const int64_t block_stride =
      sd.block_stride_elems > 0
          ? checked_mul(sd.block_stride_elems, sd.element_size,
                        "legacy block stride bytes")
          : checked_mul(sd.bs, row, "legacy tight block stride bytes");
  sd.num_planes = fmt == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS ? 2 : 1;
  for (int32_t p = 0; p < sd.num_planes; ++p) {
    sd.plane_slot_bytes[p] = row;
    sd.plane_block_stride_bytes[p] = block_stride;
  }
  return sd;
}

PreparedGroup prepare_group(const PageBufferShapeDesc& shape_desc,
                            EngineKVFormat engine_kv_format,
                            int64_t slots_per_object) {
  PageBufferShapeDesc sd = shape_desc;
  const bool separate =
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS;
  const bool fmt17 =
      engine_kv_format == EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS;
  const bool fused =
      engine_kv_format == EngineKVFormat::NL_X_NB_BS_NH_CS;
  TORCH_CHECK(separate || fmt17 || fused,
              "LMCache-Ascend block-level MP transfer currently supports "
              "NL_X_TWO_X_NB_BS_NH_HS (16), NL_X_NP_X_NB_BS_ONE_HS (17), and "
              "NL_X_NB_BS_NH_CS (13), got ",
              static_cast<int>(engine_kv_format));
  TORCH_CHECK(sd.nl > 0 && sd.nb > 0 && sd.bs > 0 && sd.nh > 0 && sd.hs > 0,
              "shape descriptor dims must be positive, got nl=", sd.nl,
              " nb=", sd.nb, " bs=", sd.bs, " nh=", sd.nh, " hs=", sd.hs);
  TORCH_CHECK(sd.element_size == 1 || sd.element_size == 2 ||
                  sd.element_size == 4,
              "element_size must be 1, 2 or 4, got ", sd.element_size);
  TORCH_CHECK(sd.kv_size == (separate ? 2 : 1),
              "kv_size must be ", (separate ? 2 : 1), " for format ",
              static_cast<int>(engine_kv_format), ", got ", sd.kv_size);

  if (sd.num_planes == 0) {
    TORCH_CHECK(!fmt17,
                "format 17 requires explicit plane geometry (num_planes > "
                "0); regenerate metadata from the real tensors");
    sd = describe_legacy_dense_token_rows(sd, engine_kv_format);
  }
  TORCH_CHECK(sd.num_planes >= 1 &&
                  sd.num_planes <= kvcache_ops::kMaxPlanes,
              "num_planes must be in [1, ", kvcache_ops::kMaxPlanes,
              "], got ", sd.num_planes);
  if (separate) {
    TORCH_CHECK(sd.num_planes == 2,
                "format 16 requires exactly 2 physical planes, got ",
                sd.num_planes);
  }
  if (fused) {
    TORCH_CHECK(sd.num_planes == 1,
                "format 13 requires exactly 1 physical plane, got ",
                sd.num_planes);
  }
  if (fmt17) {
    TORCH_CHECK(sd.nh == 1, "format 17 requires one head per plane, got nh=",
                sd.nh);
  }
  TORCH_CHECK(slots_per_object > 0,
              "slots per object must be positive, got ", slots_per_object);

  PreparedGroup group;
  group.layout.num_planes = sd.num_planes;
  const int64_t scalar_row = checked_mul(
      checked_mul(sd.nh, sd.hs, "nh * hs"), sd.element_size,
      "LMC scalar row bytes");
  int64_t prefix = 0;
  for (int32_t p = 0; p < sd.num_planes; ++p) {
    const int64_t payload = sd.plane_slot_bytes[p];
    const int64_t block_stride = sd.plane_block_stride_bytes[p];
    TORCH_CHECK(payload > 0, "plane ", p, " payload must be positive, got ",
                payload);
    TORCH_CHECK(align_up32(payload) <= kUbSegmentBytes, "plane ", p,
                " aligned row (", align_up32(payload),
                " bytes) exceeds the per-segment UB budget (", kUbSegmentBytes,
                ")");
    const int64_t span = checked_mul(sd.bs, payload, "plane block span");
    TORCH_CHECK(block_stride >= span, "plane ", p,
                " block stride (", block_stride,
                ") is below the dense block span (", span,
                "); blocks would overlap");
    TORCH_CHECK(block_stride % kGmAlignBytes == 0, "plane ", p,
                " engine block stride (", block_stride,
                " bytes) must be a multiple of ", kGmAlignBytes,
                " for DataCopy alignment");
    // Address-range overflow probe: (nb - 1) * stride + span must stay
    // inside int64.
    checked_add(checked_mul(sd.nb - 1, block_stride, "plane address range"),
                span, "plane address range");
    if (!fmt17) {
      TORCH_CHECK(payload == scalar_row, "plane ", p,
                  " payload (", payload,
                  ") must equal the full scalar row (", scalar_row,
                  ") for format ", static_cast<int>(engine_kv_format));
    }
    // LMC packed-row gap = bytes of the OTHER planes in the row; it feeds a
    // uint32 DataCopyPad field (engine-side gap is always 0: dense rows).
    TORCH_CHECK(scalar_row >= payload &&
                    scalar_row - payload <= kCopyGmGapMax,
                "plane ", p, " LMC row gap (", scalar_row - payload,
                ") exceeds the DataCopyPad GM gap range");
    group.layout.planes[p] = kvcache_ops::PlaneLayout{
        payload, block_stride, fmt17 ? prefix : 0};
    prefix = checked_add(prefix, payload, "LMC row prefix sum");
  }
  if (fmt17) {
    TORCH_CHECK(prefix == scalar_row,
                "sum of plane payloads (", prefix,
                ") must equal nh * hs * element_size (", scalar_row,
                ") for format 17");
  }

  group.layout.lmc_token_stride_bytes = scalar_row;
  group.layout.lmc_layer_stride_bytes =
      checked_mul(slots_per_object, scalar_row, "LMC layer stride bytes");
  const int64_t slab =
      checked_mul(sd.nl, group.layout.lmc_layer_stride_bytes, "LMC slab bytes");
  group.layout.lmc_object_bytes =
      checked_mul(separate ? 2 : 1, slab, "LMC object bytes");
  if (separate) {
    // 2LTD: K slab first, V slab after (planes[1] base = one full slab).
    group.layout.planes[1].lmc_base_offset_bytes = slab;
  }
  // LMC pages must not share 32B lines.
  TORCH_CHECK(checked_mul(sd.bs, scalar_row, "LMC page bytes") %
                      kGmAlignBytes ==
                  0,
              "BS * LMC row bytes (", sd.bs * scalar_row,
              ") must be a multiple of ", kGmAlignBytes,
              "; otherwise adjacent LMC pages share a 32B line");

  group.separate_plane = separate;
  group.nl = sd.nl;
  group.nb = sd.nb;
  group.bs = sd.bs;
  group.slots_per_object = slots_per_object;
  return group;
}

// ---------------------------------------------------------------------------
// validate_launch: per-transfer dynamic variables.
// ---------------------------------------------------------------------------

struct CheckedLaunch {
  int32_t total_blocks = 0;
  int32_t num_objects = 0;
  int32_t blocks_per_object = 0;
  int32_t skip_prefix_n_blocks = 0;
};

CheckedLaunch validate_launch(const PreparedGroup& group, int64_t total_blocks,
                              int num_objects, int64_t block_ids_offset,
                              int64_t block_ids_capacity,
                              int skip_prefix_n_blocks) {
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);
  TORCH_CHECK(total_blocks >= 0, "total_blocks must be non-negative, got ",
              total_blocks);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (",
              num_objects, ")");
  const int64_t blocks = total_blocks / num_objects;
  TORCH_CHECK(checked_mul(blocks, group.bs, "blocks * bs") ==
                  group.slots_per_object,
              "blocks_per_object * block_size (", blocks * group.bs,
              ") must equal slots per object (", group.slots_per_object, ")");
  TORCH_CHECK(skip_prefix_n_blocks >= 0 &&
                  skip_prefix_n_blocks <= blocks,
              "skip_prefix_n_blocks (", skip_prefix_n_blocks,
              ") must be within [0, ", blocks, "]");
  // Subtraction form: offset + length itself may not overflow.
  TORCH_CHECK(block_ids_offset >= 0 && block_ids_offset <= block_ids_capacity,
              "block_ids_offset (", block_ids_offset,
              ") must be within [0, ", block_ids_capacity, "]");
  TORCH_CHECK(total_blocks <= block_ids_capacity - block_ids_offset,
              "block_ids slice [", block_ids_offset, ", ",
              block_ids_offset + total_blocks, ") exceeds block_ids capacity ",
              block_ids_capacity);
  TORCH_CHECK(total_blocks <= std::numeric_limits<int32_t>::max() &&
                  blocks <= std::numeric_limits<int32_t>::max(),
              "block counts exceed the kernel's int32 launch arguments");

  CheckedLaunch out;
  out.total_blocks = static_cast<int32_t>(total_blocks);
  out.num_objects = num_objects;
  out.blocks_per_object = static_cast<int32_t>(blocks);
  out.skip_prefix_n_blocks = skip_prefix_n_blocks;
  return out;
}

// ---------------------------------------------------------------------------
// Launch: one kernel launch per object.
// ---------------------------------------------------------------------------

void launch_prepared_objects(uint32_t aiv_num, void* stream,
                             const PreparedGroup& group,
                             uint8_t* paged_buffer_ptrs,
                             const std::vector<int64_t>& obj_device_ptrs,
                             int64_t* block_ids_base,
                             const CheckedLaunch& launch, bool to_engine) {
  // blockDim is clamped to the work-item count so tiny transfers do not spin
  // idle cores. blocks_per_object >= 1 is guaranteed by validate_launch
  // (blocks * bs == slots_per_object > 0).
  const int32_t plane_slots =
      group.separate_plane ? group.layout.num_planes : 1;
  const int64_t work = static_cast<int64_t>(group.nl) * plane_slots *
                       launch.blocks_per_object;
  const uint32_t blockDim =
      static_cast<uint32_t>(std::min<int64_t>(aiv_num, work));
  for (int32_t i = 0; i < launch.num_objects; ++i) {
    uint8_t* engine_block_ids = reinterpret_cast<uint8_t*>(
        block_ids_base + static_cast<int64_t>(i) * launch.blocks_per_object);
    kvcache_ops::multi_layer_block_transfer_kernel(
        blockDim, stream, paged_buffer_ptrs,
        reinterpret_cast<uint8_t*>(obj_device_ptrs[i]), engine_block_ids,
        launch.blocks_per_object, launch.skip_prefix_n_blocks, group.nl,
        group.nb, group.bs, group.separate_plane, group.layout, to_engine);
  }
}

// ---------------------------------------------------------------------------
// LMC object device-VA resolution and staging (direct entry).
// ---------------------------------------------------------------------------

// CUDA UVA lets the kernel consume host data_ptr()s. Ascend DataCopy needs a
// device VA: registered aclrtMallocHost (get_device_ptr) or a true NPU
// allocation. MixedMemoryAllocator SHM L1 is neither — stage on-device and
// aclrtMemcpy.
bool is_npu_memory_ptr(void* ptr) {
  aclrtPtrAttributes attributes{};
  const aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
  if (ret != ACL_ERROR_NONE) {
    return false;
  }
  return attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
}

std::vector<int64_t> device_lmc_ptrs(const std::vector<int64_t>& ptrs) {
  std::vector<int64_t> out;
  out.reserve(ptrs.size());
  for (int64_t p : ptrs) {
    void* mapped =
        get_device_ptr(reinterpret_cast<void*>(static_cast<uintptr_t>(p)));
    out.push_back(mapped != nullptr ? reinterpret_cast<int64_t>(mapped) : p);
  }
  return out;
}

struct HostStage {
  torch::Tensor buf;
  int64_t host_ptr = 0;
  size_t nbytes = 0;
};

// Resolve LMCache object pointers to device VAs. SHM/host objects that are
// neither registered aclrtMallocHost nor NPU memory get an on-device staging
// tensor (HostStage); the caller memcpy-asyncs around the kernel. Allocations
// happen BEFORE the OpCommand so the tensors can be captured by value.
struct PreparedLmcPtrs {
  std::vector<int64_t> kernel_obj_ptrs;
  std::vector<HostStage> host_stages;
};

PreparedLmcPtrs prepare_lmc_ptrs(const std::vector<int64_t>& lmcache_objects_ptrs,
                                 const torch::Device& device,
                                 const PreparedGroup& group) {
  PreparedLmcPtrs prepared;
  prepared.kernel_obj_ptrs.reserve(lmcache_objects_ptrs.size());
  const auto staging_opts =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  // One capacity formula for every layout.
  const int64_t object_bytes = group.layout.lmc_object_bytes;
  TORCH_CHECK(object_bytes > 0, "LMCache object byte size must be positive");
  for (int64_t p : lmcache_objects_ptrs) {
    void* raw = reinterpret_cast<void*>(static_cast<uintptr_t>(p));
    void* mapped = get_device_ptr(raw);
    if (mapped != nullptr) {
      prepared.kernel_obj_ptrs.push_back(reinterpret_cast<int64_t>(mapped));
      continue;
    }
    if (is_npu_memory_ptr(raw)) {
      prepared.kernel_obj_ptrs.push_back(p);
      continue;
    }
    // A torch-pinned host pointer may not have a mapping in our registry.
    HostStage stage;
    stage.buf = torch::empty({object_bytes}, staging_opts);
    stage.host_ptr = p;
    stage.nbytes = static_cast<size_t>(object_bytes);
    prepared.kernel_obj_ptrs.push_back(
        reinterpret_cast<int64_t>(stage.buf.data_ptr()));
    prepared.host_stages.push_back(std::move(stage));
  }
  return prepared;
}

// Enqueue HostStage memcpys + the block kernel(s) on ``stream``: H2D stages
// before the kernel, D2H stages after, and a partial D2H store first reads
// the host object into staging so the untouched prefix is preserved when
// the whole object is written back.
int enqueue_block_transfer(void* stream, uint32_t aiv_num,
                           uint8_t* paged_buffer_ptrs,
                           const PreparedGroup& group,
                           const PreparedLmcPtrs& prepared,
                           int64_t* block_ids_base,
                           const CheckedLaunch& launch, bool to_engine) {
  if (to_engine || launch.skip_prefix_n_blocks > 0) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          stage.buf.data_ptr(), stage.nbytes,
          reinterpret_cast<const void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  launch_prepared_objects(aiv_num, stream, group, paged_buffer_ptrs,
                          prepared.kernel_obj_ptrs, block_ids_base, launch,
                          to_engine);
  if (!to_engine) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          reinterpret_cast<void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, stage.buf.data_ptr(), stage.nbytes,
          ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  return 0;
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Static geometry (prepare_group) + dynamic variables (validate_launch)
  const PreparedGroup group =
      prepare_group(shape_desc, engine_kv_format, lmcache_chunk_size);
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  const int64_t total_blocks = block_ids.size(0);
  const CheckedLaunch launch =
      validate_launch(group, total_blocks, num_objects,
                      /*block_ids_offset=*/0,
                      /*block_ids_capacity=*/total_blocks,
                      skip_prefix_n_blocks);

  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == at::kLong,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_privateuseone(),
              "paged_buffer_ptrs_tensor must live on the NPU");
  TORCH_CHECK(paged_buffer_ptrs_tensor.dim() == 1,
              "paged_buffer_ptrs_tensor must be one-dimensional");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_contiguous(),
              "paged_buffer_ptrs_tensor must be contiguous");
  // Pointer table: one device pointer per (layer, physical plane).
  const int64_t expected_ptrs =
      static_cast<int64_t>(group.layout.num_planes) * group.nl;
  TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == expected_ptrs,
              "paged_buffer_ptrs_tensor must contain num_planes * nl "
              "pointers: expected ",
              expected_ptrs, ", got ", paged_buffer_ptrs_tensor.numel());
  TORCH_CHECK(block_ids.is_privateuseone(), "block_ids must live on the NPU");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong,
              "block_ids must have dtype int64");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be one-dimensional");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");

  const bool to_engine = (direction == TransferDirection::H2D);

  const c10::OptionalDeviceGuard device_guard(device);
  PreparedLmcPtrs prepared = prepare_lmc_ptrs(lmcache_objects_ptrs, device, group);

  uint8_t* paged_buffer_ptrs =
      static_cast<uint8_t*>(paged_buffer_ptrs_tensor.data_ptr());
  int64_t* block_ids_base = block_ids.data_ptr<int64_t>();

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  cmd.SetCustomHandler([stream, paged_buffer_ptrs, group, prepared,
                        block_ids_base, launch, to_engine]() -> int {
    const char* socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    return enqueue_block_transfer(stream, aiv_num, paged_buffer_ptrs, group,
                                  prepared, block_ids_base, launch, to_engine);
  });
  cmd.Run();
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src, size_t nbytes,
                                    TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream);

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in
  // order (mirrors upstream execute_object_group_transfer).
  const c10::OptionalDeviceGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  TORCH_CHECK(device.is_privateuseone(), "device must be an NPU device");
  TORCH_CHECK(host_buffer_alignment > 0 &&
                  (host_buffer_alignment & (host_buffer_alignment - 1)) == 0,
              "host_buffer_alignment must be a non-zero power of two, got ",
              host_buffer_alignment);

  // --- Whole-plan pre-validation: every group is prepared and
  // every launch is validated before ANY staging copy or kernel launch is
  // enqueued. A failure here means "nothing started".
  std::vector<PreparedGroup> groups;
  groups.reserve(kernel_group_specs.size());
  for (const auto& spec : kernel_group_specs) {
    groups.push_back(prepare_group(spec.shape_desc, spec.engine_kv_format,
                                   spec.lmcache_chunk_size));
  }

  struct PreparedLaunch {
    int32_t group_idx;
    CheckedLaunch checked;
    int64_t block_ids_offset;
  };
  std::vector<std::vector<PreparedLaunch>> prepared_steps;
  prepared_steps.reserve(batch_steps.size());
  for (const auto& step : batch_steps) {
    for (const auto& copy : step.staging) {
      // Raw external pointers carry no derivable allocation capacity; the
      // plan builder owns pointer/size validity. What can be
      // checked cheaply here is checked.
      TORCH_CHECK(copy.nbytes > 0, "StagingCopy nbytes must be positive");
      TORCH_CHECK(copy.dest != 0 && copy.src != 0,
                  "StagingCopy pointers must be non-null");
    }
    std::vector<PreparedLaunch> launches;
    launches.reserve(step.launches.size());
    for (const auto& launch : step.launches) {
      TORCH_CHECK(launch.group_idx >= 0 &&
                      launch.group_idx <
                          static_cast<int>(kernel_group_specs.size()),
                  "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& spec = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects <=
                      static_cast<int>(spec.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  spec.lmcache_objects_ptrs.size(), ")");
      PreparedLaunch prepared;
      prepared.group_idx = launch.group_idx;
      prepared.checked =
          validate_launch(groups[launch.group_idx], launch.total_blocks,
                          launch.num_objects, launch.block_ids_offset,
                          spec.block_ids_capacity, launch.skip_prefix_n_blocks);
      prepared.block_ids_offset = launch.block_ids_offset;
      launches.push_back(prepared);
    }
    prepared_steps.push_back(std::move(launches));
  }

  const char* socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  // Capture by value like multi_layer_block_kv_transfer: OpCommand may run
  // the handler on a worker thread; [&] deadlocks torch_npu's task queue.
  // The pointer table, block IDs, staging buffers and engine plane views are
  // owned by the caller's cache context, which synchronizes the stream before
  // releasing them, so the captured raw pointers stay valid.
  cmd.SetCustomHandler([direction, is_h2d, host_buffer_alignment, aiv_num,
                        stream, kernel_group_specs, batch_steps, groups,
                        prepared_steps]() -> int {
    const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
      for (const auto& copy : staging) {
        lmcache_memcpy_async_on_stream(copy.dest, copy.src, copy.nbytes,
                                       direction, copy.host_offset,
                                       host_buffer_alignment, stream);
      }
    };

    for (size_t step_idx = 0; step_idx < batch_steps.size(); ++step_idx) {
      // H2D stages CPU->NPU temp buffers before the kernel reads them; D2H
      // stages NPU->CPU after the kernel writes them. The per-step ordering
      // must be preserved because temp buffers are reused across steps.
      if (is_h2d) {
        do_staging(batch_steps[step_idx].staging);
      }
      for (const auto& launch : prepared_steps[step_idx]) {
        const KernelGroupSpec& spec = kernel_group_specs[launch.group_idx];
        const PreparedGroup& group = groups[launch.group_idx];
        std::vector<int64_t> obj_device_ptrs = device_lmc_ptrs(
            std::vector<int64_t>(
                spec.lmcache_objects_ptrs.begin(),
                spec.lmcache_objects_ptrs.begin() +
                    launch.checked.num_objects));
        int64_t* block_ids_base = reinterpret_cast<int64_t*>(
            spec.block_ids_base +
            static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t));
        launch_prepared_objects(
            aiv_num, stream, group,
            reinterpret_cast<uint8_t*>(spec.paged_buffer_ptrs),
            obj_device_ptrs, block_ids_base, launch.checked, is_h2d);
      }
      if (!is_h2d) {
        do_staging(batch_steps[step_idx].staging);
      }
    }
    return 0;
  });
  cmd.Run();
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src, size_t nbytes,
                                    TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK(host_buffer_alignments > 0 &&
                  (host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be a non-zero power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  const aclrtMemcpyKind kind = (direction == TransferDirection::H2D)
                                    ? ACL_MEMCPY_HOST_TO_DEVICE
                                    : ACL_MEMCPY_DEVICE_TO_HOST;

  // Split the copy at the host buffer's alignment boundaries: each chunk
  // stays inside one aligned region (port of upstream lmcache_memcpy_async).
  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    const size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    const size_t max_nbytes = real_end - offset - host_buffer_offset;

    const aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void*>(dest + offset), max_nbytes,
        reinterpret_cast<const void*>(src + offset), max_nbytes, kind, stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMemcpyAsync failed: ret=", ret);

    offset += max_nbytes;
  }
}

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  lmcache_memcpy_async_on_stream(
      dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments,
      c10_npu::getCurrentNPUStream().stream());
}
