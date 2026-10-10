# Block-Level KV Transfer Kernel Contract

This document is the host/device contract for the MP block-level KV transfer:

- device kernel: `multi_layer_block_mem_kernels.cpp` 
- host parser/validator: `csrc/mp_mem_kernels.cpp`
  (`prepare_group` + `validate_launch` are the single admission authority)

The kernel is a pure byte mover: every geometry question (format, plane
packing, strides, offsets) is resolved host-side and passed in as a flat
`BlockTransferLayout` POD. Everything below must hold before launch; the
host layer enforces it and rejects violations with `TORCH_CHECK`.

## 1. Supported engine KV formats

| Format | Enum | `separate_plane` | Work item | LMC layout |
|---|---|---|---|---|
| 16 | `NL_X_TWO_X_NB_BS_NH_HS` | `true` | (layer, plane, block) | `[2, L, T, NH*HS]`, K slab then V slab |
| 17 | `NL_X_NP_X_NB_BS_ONE_HS` | `false` | (layer, block), all planes moved by one core | `[L, T, sum(payload)]` packed byte rows |

Any other format is rejected by `prepare_group`. With
`separate_plane == false`, every plane of a page is moved by ONE core so no
two cores ever share a 32B LMC line (thin planes such as a 2B scale share
cache lines with their row neighbours).

## 2. Host-side validation (launch gate)

Geometry (`prepare_group`, resolved once per group):

- `nl, nb, bs, nh, hs > 0`; `element_size in {1, 2, 4}`
- `kv_size == 2` for format 16, `== 1` for format 17
- `1 <= num_planes <= 4`; exactly `2` for format 16; `nh == 1` for format 17
- per plane: `payload_bytes > 0` and
  `AlignUp32(payload_bytes) <= ub_segment` (one aligned row of every plane
  fits one queue slot), where `ub_segment` is the host-queried UB budget
  divided by the queue depth (see section 3)
- per plane: `engine_block_stride_bytes >= bs * payload_bytes` (no block
  overlap) and `engine_block_stride_bytes % 32 == 0`
- address-range probe `(nb - 1) * block_stride + bs * payload` fits int64
- format 16: `payload_bytes == nh * hs * element_size` per plane
- format 17: `sum(payload_bytes) == nh * hs * element_size`
  (the packed row width)
- `bs * lmc_row_bytes % 32 == 0` (adjacent LMC pages do not share a 32B line)
- `lmc_row_bytes - payload_bytes <= uint32 max`
  (DataCopyPad GM-side gap field is uint32)

Launch variables (`validate_launch`, per transfer):

- `1 <= num_objects <= 4`; `total_blocks % num_objects == 0`
- `blocks_per_object * bs == slots_per_object == lmcache_chunk_size`
- `0 <= skip_prefix_n_blocks <= blocks_per_object`
- the `block_ids` slice fits the provided capacity; counts fit int32

Producer contract (runtime, not launch-checkable):

- every `block_ids[i]` is in `[0, nb)`. The kernel defensively skips
  out-of-range ids -- a data-dependent check that, unlike `skip`, cannot be
  folded into the launch because the id values live in GM.

## 3. UB budget and its relation to block size

`kBlockTransferUbBytes` (128KB) bounds a single token ROW, not a whole
block. The kernel derives its segment size as:

```
tokens_per_segment = min(bs,
                         kBlockTransferUbBytes
                             / (kBlockTransferQueueDepth
                                * AlignUp32(max payload_bytes)),
                         kMaxRowsPerDataCopyPad)
```

- `bs` (tokens per block) is UNCONSTRAINED: rows are moved in segments of
  `tokens_per_segment` and one plane block may span multiple segments
- The only hard geometry requirement is that one aligned row of the widest
  plane fits a single queue slot (validated in `prepare_group`).
- The host queries the real size at launch time via
  `PlatformAscendCManager::GetCoreMemSize(CoreMemType::UB)` and passes it
  down verbatim as `ub_bytes`. `ub_bytes == 0` (query unavailable) falls 
  back to the built-in 128KB floor. 

## 4. DataCopyPad limits

Per the AscendC API reference for `DataCopyExtParams`:

| Field | Range | Kernel usage |
|---|---|---|
| `blockCount` | [1, 4095] | rows per segment (clamped to `kMaxRowsPerDataCopyPad`) |
| `blockLen` | [1, 2097151] bytes | `payload_bytes` per row |
| `srcStride` / `dstStride` | uint32 | GM side in bytes, UB side in 32B blocks |

Alignment: `DataCopyPad` has NO GM-side address alignment constraint; the
UB-side `LocalTensor` start must be 32B aligned (guaranteed by the queue
allocator). The contiguous fast path uses plain `DataCopy` instead and
therefore additionally requires 32B-aligned GM bases and a 32B-multiple
payload; the kernel checks this per work item and falls back to the row
path otherwise.

## 5. Work distribution and skip semantics

- `block_dim = min(AIV cores, ACTIVE work items)`
  `= min(cores, nl * plane_slots * (blocks - skip))`.
- Work items span ACTIVE blocks only: `block = skip + (w % active_blocks)`.
  No core iterates over skipped blocks; there are no dead work items.
- `skip == blocks` (a legal input) means nothing to transfer: the host
  skips the launch entirely (`<<<0>>>` would be an invalid launch) and the
  kernel-side loop is empty.
- Out-of-range engine block ids are skipped at runtime (data-dependent, see
  the producer contract above).

## 6. Pointer table

`ptrs[layer * num_planes + plane]`: each entry is the plane tensor view's
`data_ptr()`, already including the view's `storage_offset`. The table must
live on the NPU and contain exactly `nl * num_planes` int64 entries.
