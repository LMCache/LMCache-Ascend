/*
 * Minimal forward declarations from hcomm (open-source: https://gitee.com/ascend/hcomm)
 * to call MemMappingManager::GetDevVA without the full hcomm header tree.
 *
 * Relevant upstream headers:
 *   - include/hccl/hccl_types.h   (HcclResult enum)
 *   - pkg_inc/hccl/base.h         (u64, s32 typedefs)
 *   - src/platform/inc/mem_mapping_manager.h  (MemMappingManager class)
 *
 * The symbols are resolved at link time from libhccl.so.
 * This is an ABI contract — update if the upstream signatures change.
 */

#include "hcomm_devva.h"

/* ---- type aliases (hcomm/pkg_inc/hccl/base.h) ---- */
typedef signed int     s32;
typedef unsigned long long u64;

/* ---- HcclResult (hcomm/include/hccl/hccl_types.h) ---- */
typedef enum {
    HCCL_SUCCESS   = 0,
    HCCL_E_INTERNAL = 4,
} HcclResult;

/* ---- MemMappingManager forward declaration ---- */
namespace hccl {
class MemMappingManager {
public:
    static MemMappingManager &GetInstance(s32 deviceLogicID);
    HcclResult GetDevVA(s32 deviceLogicID, void *addr, u64 size, void *&devVA);
};
}  // namespace hccl

/* ---- C wrapper implementation ---- */
extern "C" int hcomm_get_dev_va(int device_logic_id,
                                void *host_ptr,
                                uint64_t size,
                                void **dev_va)
{
    void *devVA = nullptr;
    HcclResult ret = hccl::MemMappingManager::GetInstance(
                         static_cast<s32>(device_logic_id))
                         .GetDevVA(static_cast<s32>(device_logic_id),
                                   host_ptr,
                                   static_cast<u64>(size),
                                   devVA);
    if (ret != HCCL_SUCCESS) {
        return static_cast<int>(ret);
    }
    *dev_va = devVA;
    return 0;
}
