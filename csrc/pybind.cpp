// SPDX-License-Identifier: Apache-2.0

#include "cachegen_kernels.h"
#include "cb_retrieve_plan.h"
#include "completion_recorder.h"
#include "dcmi_management.h"
#include "event_recorder.h"
#include "managed_mem.h"
#include "mem_alloc.h"
#include "mem_kernels.h"
#include "mp_mem_kernels.h"
#include "pac_kernels.h"
#include "pos_kernels.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

namespace py = pybind11;

std::vector<torch::Tensor> normalize_kv_caches(const py::object &input) {
  if (THPVariable_Check(input.ptr())) {
    return {input.cast<torch::Tensor>()};
  } else if (py::isinstance<py::tuple>(input)) {
    return input.cast<std::vector<torch::Tensor>>();
  } else {
    throw std::runtime_error(
        "vllm_kv_caches must be a Tensor or a tuple of Tensors");
  }
}

namespace {
constexpr size_t kMaxPlanes = 4;

py::tuple plane_array_get(const int64_t *arr, int32_t n) {
  if (n <= 0) {
    return py::tuple();
  }
  const size_t len =
      static_cast<size_t>(n) < kMaxPlanes ? static_cast<size_t>(n) : kMaxPlanes;
  py::tuple out(len);
  for (size_t i = 0; i < len; ++i) {
    out[i] = arr[i];
  }
  return out;
}

void plane_array_set(int64_t *arr, const std::vector<int64_t> &v,
                     const char *name) {
  if (v.size() > kMaxPlanes) {
    throw py::value_error(std::string(name) + " length exceeds max 4");
  }
  for (size_t i = 0; i < kMaxPlanes; ++i) {
    arr[i] = i < v.size() ? v[i] : 0;
  }
}
} // namespace

void single_layer_kv_transfer_wrapper(torch::Tensor &lmc_key_value_cache,
                                      const py::object &vllm_kv_caches_obj,
                                      torch::Tensor &slot_mapping,
                                      bool direction, int kvcache_format_raw,
                                      bool token_major, bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  single_layer_kv_transfer(lmc_key_value_cache, vllm_kv_caches, slot_mapping,
                           direction, kvcache_format_raw, token_major,
                           vllm_two_major);
}

void batched_fused_single_layer_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    const py::object &vllm_kv_caches_obj, torch::Tensor &slot_mapping_full,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    bool direction, int kvcache_format_raw, bool token_major,
    bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  batched_fused_single_layer_kv_transfer(
      lmc_tensors, staging_cache, vllm_kv_caches, slot_mapping_full,
      chunk_offsets, chunk_sizes, direction, kvcache_format_raw, token_major,
      vllm_two_major);
}

PYBIND11_MODULE(c_ops, m) {
  m.def("get_device_ptr", [](uintptr_t ptr_addr) {
    return reinterpret_cast<uintptr_t>(
        get_device_ptr(reinterpret_cast<void *>(ptr_addr)));
  });
  m.def("register_mapping",
        [](uintptr_t host_ptr, uintptr_t dev_ptr, size_t size) {
          return reinterpret_cast<uintptr_t>(
              register_mapping(reinterpret_cast<void *>(host_ptr),
                               reinterpret_cast<void *>(dev_ptr), size));
        });
  m.def("unregister_ptr", [](uintptr_t ptr_addr) {
    return unregister_ptr(reinterpret_cast<void *>(ptr_addr));
  });
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("fused_multi_layer_kv_transfer", &fused_multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_310p", &multi_layer_kv_transfer_310p);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer_wrapper);
  m.def("batched_fused_single_layer_kv_transfer",
        &batched_fused_single_layer_kv_transfer_wrapper);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_ascend_new);
  m.def("decode_fast_new", &decode_ascend_new);
  m.def("decode_fast_prefsum", &decode_ascend_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);

  m.def("pac_prepare_enc_metadata", &pac_prepare_enc_metadata);
  m.def("pac_encode", &pac_encode);
  m.def("pac_decode", &pac_decode);

  m.def("record_event_on_stream", &record_event_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("event_type_name"),
        py::arg("session_id"), py::arg("str_metadata"), py::arg("int_metadata"),
        py::call_guard<py::gil_scoped_release>());
  m.def("drain_recorded_events", &drain_recorded_events);
  m.def("record_completion_on_stream", &record_completion_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("kind"), py::arg("payload"),
        py::call_guard<py::gil_scoped_release>());
  // Return each payload as py::bytes; pybind11 utf-8-decodes std::string
  // by default, corrupting binary payloads (e.g. msgpack).
  m.def("drain_recorded_completions", []() {
    auto items = drain_recorded_completions();
    py::list out;
    for (auto &kv : items) {
      out.append(py::make_tuple(py::str(kv.first), py::bytes(kv.second)));
    }
    return out;
  });

  py::enum_<TransferDirection>(m, "TransferDirection", py::module_local())
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<EngineKVFormat>(m, "EngineKVFormat", py::module_local())
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS)
      .value("NL_X_TWO_NB_NH_ONE_BS_HS",
             EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS)
      .value("NL_X_TWO_X_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS)
      .value("NL_X_NP_X_NB_BS_ONE_HS", EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS)
      .export_values();
  // Match upstream lmcache_native / CUDA pybind: tests and connectors use
  // GPUKVFormat as an alias of EngineKVFormat.
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");
  // Factory constructs this class via device_ops.PageBufferShapeDesc()
  // after NpuDeviceOps.bind_native. Kernels take it by value (same as CUDA
  // + lmcache_native). dynamic_attr keeps torch dtype for the Python
  // torch_ops fallback.
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc", py::module_local(),
                                  py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems)
      .def_readwrite("num_planes", &PageBufferShapeDesc::num_planes)
      .def_property(
          "plane_slot_bytes",
          [](const PageBufferShapeDesc &s) {
            return plane_array_get(s.plane_slot_bytes, s.num_planes);
          },
          [](PageBufferShapeDesc &s, const std::vector<int64_t> &v) {
            plane_array_set(s.plane_slot_bytes, v, "plane_slot_bytes");
          })
      .def_property(
          "plane_block_stride_bytes",
          [](const PageBufferShapeDesc &s) {
            return plane_array_get(s.plane_block_stride_bytes, s.num_planes);
          },
          [](PageBufferShapeDesc &s, const std::vector<int64_t> &v) {
            plane_array_set(s.plane_block_stride_bytes, v,
                            "plane_block_stride_bytes");
          });
  m.def(
      "multi_layer_block_kv_transfer",
      [](const torch::Tensor &paged_buffer_ptrs_tensor,
         std::vector<int64_t> lmcache_objects_ptrs,
         const torch::Tensor &block_ids, const torch::Device &device,
         const py::object &direction, PageBufferShapeDesc shape_desc,
         int lmcache_chunk_size, const py::object &engine_kv_format,
         int skip_prefix_n_blocks) {
        const auto dir =
            static_cast<TransferDirection>(py::int_(direction).cast<int>());
        const auto fmt =
            static_cast<EngineKVFormat>(py::int_(engine_kv_format).cast<int>());
        // Direct launch path: release the GIL while the transfer drains
        // torch_npu's ACL task queue (getCurrentNPUStream().stream()) and
        // enqueues kernels + memcpys. A c10::Error from TORCH_CHECK
        // re-acquires the GIL during unwinding, so failures still surface as
        // Python exceptions (mp worker threads keep a clean error contract).
        py::gil_scoped_release release;
        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, dir, shape_desc, lmcache_chunk_size, fmt,
            skip_prefix_n_blocks);
      },
      py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
      py::arg("block_ids"), py::arg("device"), py::arg("direction"),
      py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
      py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"));
  py::class_<StagingCopy>(m, "StagingCopy", py::module_local())
      .def(py::init([](uintptr_t dest, uintptr_t src, size_t nbytes,
                       size_t host_offset) {
             return StagingCopy{dest, src, nbytes, host_offset};
           }),
           py::arg("dest"), py::arg("src"), py::arg("nbytes"),
           py::arg("host_offset"));
  py::class_<LaunchVar>(m, "LaunchVar", py::module_local())
      .def(
          py::init([](int group_idx, int64_t block_ids_offset, int total_blocks,
                      int num_objects, int skip_prefix_n_blocks) {
            return LaunchVar{group_idx, block_ids_offset, total_blocks,
                             num_objects, skip_prefix_n_blocks};
          }),
          py::arg("group_idx"), py::arg("block_ids_offset"),
          py::arg("total_blocks"), py::arg("num_objects"),
          py::arg("skip_prefix_n_blocks"));
  py::class_<BatchStep>(m, "BatchStep", py::module_local())
      .def(py::init([](std::vector<StagingCopy> staging,
                       std::vector<LaunchVar> launches) {
             return BatchStep{std::move(staging), std::move(launches)};
           }),
           py::arg("staging"), py::arg("launches"));
  py::class_<KernelGroupSpec>(m, "KernelGroupSpec", py::module_local())
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
                       int engine_kv_format, uintptr_t block_ids_base,
                       int64_t block_ids_capacity) {
             return KernelGroupSpec{
                 paged_buffer_ptrs,
                 std::move(lmcache_objects_ptrs),
                 std::move(shape_desc),
                 lmcache_chunk_size,
                 static_cast<EngineKVFormat>(engine_kv_format),
                 block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));
  m.def(
      "execute_object_group_transfer",
      [](int direction, const torch::Device &device,
         size_t host_buffer_alignment,
         const std::vector<KernelGroupSpec> &kernel_group_specs,
         const std::vector<BatchStep> &batch_steps) {
        return execute_object_group_transfer(
            static_cast<TransferDirection>(direction), device,
            host_buffer_alignment, kernel_group_specs, batch_steps);
      },
      py::arg("direction"), py::arg("device"), py::arg("host_buffer_alignment"),
      py::arg("kernel_group_specs"), py::arg("batch_steps"),
      // Direct launch path (CUDA parity): every kernel launch and memcpy is
      // enqueued on the calling thread after draining torch_npu's ACL task
      // queue (getCurrentNPUStream().stream()), so the GIL can be released
      // for the whole enqueue.
      py::call_guard<py::gil_scoped_release>());

  // Field-for-field ABI match with LMCache's CUDA CacheBlend planner. Only
  // slot_mapping_* changes between requests; all other fields describe the
  // registered paged cache and its temporary slots.
  py::class_<CBGroupSpec>(m, "CBGroupSpec", py::module_local())
      .def(
          py::init(
              [](uintptr_t paged_kv_ptrs, std::vector<int64_t> temp_buffer_ptrs,
                 int num_layers, int slot_tokens, int hidden_elems,
                 int element_size, int engine_kv_format, int page_buffer_size,
                 int block_size, int head_size, uintptr_t slot_mapping_base,
                 int64_t slot_mapping_capacity, uintptr_t cos_sin_cache,
                 int rot_dim, int rope_num_kv_heads, int64_t rope_head_stride,
                 int key_scalar_type, bool is_neox, int64_t rope_base_offset,
                 int64_t block_stride_elems, int cache_scalar_type) {
                return CBGroupSpec{
                    paged_kv_ptrs,
                    std::move(temp_buffer_ptrs),
                    num_layers,
                    slot_tokens,
                    hidden_elems,
                    element_size,
                    static_cast<EngineKVFormat>(engine_kv_format),
                    page_buffer_size,
                    block_size,
                    head_size,
                    block_stride_elems,
                    slot_mapping_base,
                    slot_mapping_capacity,
                    cos_sin_cache,
                    rot_dim,
                    rope_num_kv_heads,
                    rope_head_stride,
                    key_scalar_type,
                    cache_scalar_type < 0 ? key_scalar_type : cache_scalar_type,
                    is_neox,
                    rope_base_offset};
              }),
          py::arg("paged_kv_ptrs"), py::arg("temp_buffer_ptrs"),
          py::arg("num_layers"), py::arg("slot_tokens"),
          py::arg("hidden_elems"), py::arg("element_size"),
          py::arg("engine_kv_format"), py::arg("page_buffer_size"),
          py::arg("block_size"), py::arg("head_size"),
          py::arg("slot_mapping_base"), py::arg("slot_mapping_capacity"),
          py::arg("cos_sin_cache"), py::arg("rot_dim"),
          py::arg("rope_num_kv_heads"), py::arg("rope_head_stride"),
          py::arg("key_scalar_type"), py::arg("is_neox"),
          py::arg("rope_base_offset") = 0, py::arg("block_stride_elems") = 0,
          py::arg("cache_scalar_type") = -1)
      .def_readwrite("slot_mapping_base", &CBGroupSpec::slot_mapping_base)
      .def_readwrite("slot_mapping_capacity",
                     &CBGroupSpec::slot_mapping_capacity);

  // Flat int64 tables avoid constructing several Python objects per chunk:
  // staging/scatters/ropes have four columns and step_offsets contains the
  // cumulative exclusive row end for those three tables at every step.
  m.def(
      "execute_cb_retrieve_plan_flat",
      [](const torch::Device &device, size_t host_buffer_alignment,
         const std::vector<CBGroupSpec> &group_specs,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             staging,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast> ropes,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             scatters,
         py::array_t<int64_t, py::array::c_style | py::array::forcecast>
             step_offsets) {
        TORCH_CHECK(staging.ndim() == 2 && staging.shape(1) == 4,
                    "staging table must be [n, 4]");
        TORCH_CHECK(ropes.ndim() == 2 && ropes.shape(1) == 4,
                    "ropes table must be [n, 4]");
        TORCH_CHECK(scatters.ndim() == 2 && scatters.shape(1) == 4,
                    "scatters table must be [n, 4]");
        TORCH_CHECK(step_offsets.ndim() == 2 && step_offsets.shape(1) == 3,
                    "step_offsets table must be [n_steps, 3]");

        const int64_t *st = staging.data();
        const int64_t *rp = ropes.data();
        const int64_t *sc = scatters.data();
        const int64_t *so = step_offsets.data();
        const int64_t n_steps = step_offsets.shape(0);
        const auto checked_int = [](int64_t value, const char *name) {
          TORCH_CHECK(value >= std::numeric_limits<int>::min() &&
                          value <= std::numeric_limits<int>::max(),
                      name, " is outside the C++ int range: ", value);
          return static_cast<int>(value);
        };

        std::vector<CBRetrieveStep> steps;
        steps.reserve(n_steps);
        int64_t s0 = 0;
        int64_t r0 = 0;
        int64_t c0 = 0;
        for (int64_t i = 0; i < n_steps; ++i) {
          const int64_t s1 = so[i * 3];
          const int64_t r1 = so[i * 3 + 1];
          const int64_t c1 = so[i * 3 + 2];
          TORCH_CHECK(s1 >= s0 && s1 <= staging.shape(0) && r1 >= r0 &&
                          r1 <= ropes.shape(0) && c1 >= c0 &&
                          c1 <= scatters.shape(0),
                      "step_offsets not monotone or out of range at step ", i);

          CBRetrieveStep step;
          step.staging.reserve(s1 - s0);
          for (int64_t j = s0; j < s1; ++j) {
            TORCH_CHECK(st[j * 4] > 0 && st[j * 4 + 1] > 0,
                        "staging pointers must be positive at row ", j);
            TORCH_CHECK(st[j * 4 + 2] > 0 && st[j * 4 + 3] >= 0,
                        "staging nbytes/host_offset invalid at row ", j);
            step.staging.push_back(
                StagingCopy{static_cast<uintptr_t>(st[j * 4]),
                            static_cast<uintptr_t>(st[j * 4 + 1]),
                            static_cast<size_t>(st[j * 4 + 2]),
                            static_cast<size_t>(st[j * 4 + 3])});
          }
          step.ropes.reserve(r1 - r0);
          for (int64_t j = r0; j < r1; ++j) {
            step.ropes.push_back(
                CBRopeVar{checked_int(rp[j * 4], "rope group_idx"),
                          checked_int(rp[j * 4 + 1], "rope slot_idx"),
                          rp[j * 4 + 2], rp[j * 4 + 3]});
          }
          step.scatters.reserve(c1 - c0);
          for (int64_t j = c0; j < c1; ++j) {
            step.scatters.push_back(CBScatterVar{
                checked_int(sc[j * 4], "scatter group_idx"),
                checked_int(sc[j * 4 + 1], "scatter slot_idx"), sc[j * 4 + 2],
                checked_int(sc[j * 4 + 3], "scatter n_tok")});
          }
          steps.push_back(std::move(step));
          s0 = s1;
          r0 = r1;
          c0 = c1;
        }
        TORCH_CHECK(s0 == staging.shape(0) && r0 == ropes.shape(0) &&
                        c0 == scatters.shape(0),
                    "step_offsets must consume every plan table row");
        {
          py::gil_scoped_release release;
          execute_cb_retrieve_plan(device, host_buffer_alignment, group_specs,
                                   steps);
        }
      },
      py::arg("device"), py::arg("host_buffer_alignment"),
      py::arg("group_specs"), py::arg("staging"), py::arg("ropes"),
      py::arg("scatters"), py::arg("step_offsets"));
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
}
