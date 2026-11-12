// SPDX-License-Identifier: Apache-2.0

#include "multi_layer_token_mem_kernels.h"
#include "kernel_operator.h"

namespace {

class MultiLayerTokenKVTransfer {
public:
  __aicore__ inline void Init(GM_ADDR pointers, GM_ADDR temp,
                              GM_ADDR slot_mapping, int32_t num_tokens,
                              kvcache_ops::TokenKVTransferLayout layout,
                              bool page_to_lmc, AscendC::TPipe* pipe) {
    paged_kv_ptrs_ = reinterpret_cast<__gm__ uint64_t*>(pointers);
    lmc_buffer_ = reinterpret_cast<__gm__ uint8_t*>(temp);
    slot_mapping_ = reinterpret_cast<__gm__ int64_t*>(slot_mapping);
    num_layers_ = layout.num_layers;
    slot_tokens_ = layout.slot_tokens;
    num_tokens_ = num_tokens;
    hidden_bytes_ = layout.row_bytes;
    page_buffer_size_ = layout.page_buffer_size;
    block_size_ = layout.block_size;
    block_stride_bytes_ = layout.block_stride_bytes;
    one_pointer_per_layer_ = layout.one_pointer_per_layer;
    page_to_lmc_ = page_to_lmc;
    pipe->InitBuffer(queue_, 1, hidden_bytes_);
  }

  __aicore__ inline void Process() {
    const int64_t work = static_cast<int64_t>(num_layers_) * 2 * num_tokens_;
    for (int64_t w = AscendC::GetBlockIdx(); w < work;
         w += AscendC::GetBlockNum()) {
      const int32_t token = static_cast<int32_t>(w % num_tokens_);
      const int32_t plane = static_cast<int32_t>((w / num_tokens_) % 2);
      const int32_t layer =
          static_cast<int32_t>(w / (static_cast<int64_t>(num_tokens_) * 2));
      const int64_t slot = slot_mapping_[token];
      if (slot < 0 || slot >= page_buffer_size_) {
        continue;
      }

      const int64_t block = slot / block_size_;
      const int64_t in_block = slot % block_size_;
      const int64_t pointer_idx = one_pointer_per_layer_
                                      ? layer
                                      : static_cast<int64_t>(layer) * 2 + plane;
      auto* plane_base =
          reinterpret_cast<__gm__ uint8_t*>(paged_kv_ptrs_[pointer_idx]);
      const int64_t plane_offset =
          one_pointer_per_layer_
              ? static_cast<int64_t>(plane) * block_size_ * hidden_bytes_
              : 0;
      auto* engine_ptr = plane_base + block * block_stride_bytes_ +
                         plane_offset + in_block * hidden_bytes_;

      const int64_t temp_offset =
          ((static_cast<int64_t>(plane) * num_layers_ + layer) * slot_tokens_ +
           token) *
          hidden_bytes_;
      AscendC::GlobalTensor<uint8_t> temp_row;
      AscendC::GlobalTensor<uint8_t> engine_row;
      temp_row.SetGlobalBuffer(lmc_buffer_ + temp_offset, hidden_bytes_);
      engine_row.SetGlobalBuffer(engine_ptr, hidden_bytes_);

      auto ub = queue_.AllocTensor<uint8_t>();
      AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(hidden_bytes_),
                                      0u, 0u, 0u};
      AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
      AscendC::DataCopyPad(ub, page_to_lmc_ ? engine_row : temp_row, copy, pad);
      queue_.EnQue(ub);
      ub = queue_.DeQue<uint8_t>();
      AscendC::DataCopyPad(page_to_lmc_ ? temp_row : engine_row, ub, copy);
      queue_.FreeTensor(ub);
    }
  }

private:
  __gm__ uint64_t* paged_kv_ptrs_;
  __gm__ uint8_t* lmc_buffer_;
  __gm__ int64_t* slot_mapping_;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    1>
      queue_;
  bool one_pointer_per_layer_, page_to_lmc_;
  int32_t num_layers_, slot_tokens_, num_tokens_, block_size_;
  int64_t hidden_bytes_, page_buffer_size_, block_stride_bytes_;
};

} // namespace

extern "C" __global__ __aicore__ void multi_layer_token_kv_transfer_kernel(
    GM_ADDR pointers, GM_ADDR lmc_buffer, GM_ADDR slot_mapping,
    int32_t num_tokens, int32_t num_layers, int32_t slot_tokens,
    int32_t block_size, int64_t page_buffer_size, int64_t row_bytes,
    int64_t block_stride_bytes, bool one_pointer_per_layer, bool page_to_lmc) {
  const kvcache_ops::TokenKVTransferLayout layout{
      num_layers, slot_tokens,        block_size,           page_buffer_size,
      row_bytes,  block_stride_bytes, one_pointer_per_layer};
  AscendC::TPipe pipe;
  MultiLayerTokenKVTransfer op;
  op.Init(pointers, lmc_buffer, slot_mapping, num_tokens, layout, page_to_lmc,
          &pipe);
  op.Process();
}

namespace kvcache_ops {

void multi_layer_token_kv_transfer_kernel(
    uint32_t block_dim, void* stream, uint8_t* paged_kv_ptrs,
    uint8_t* lmc_buffer, uint8_t* slot_mapping, int32_t num_tokens,
    TokenKVTransferLayout layout, bool page_to_lmc) {
  ::multi_layer_token_kv_transfer_kernel<<<block_dim, nullptr, stream>>>(
      paged_kv_ptrs, lmc_buffer, slot_mapping, num_tokens, layout.num_layers,
      layout.slot_tokens, layout.block_size, layout.page_buffer_size,
      layout.row_bytes, layout.block_stride_bytes, layout.one_pointer_per_layer,
      page_to_lmc);
}

} // namespace kvcache_ops
