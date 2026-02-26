/*
 * C-linkage wrapper to query device VA from hcomm's MemMappingManager.
 * Implementation in hcomm_devva.cc; symbols resolved from libhccl.so.
 */
#ifndef HCOMM_DEVVA_H
#define HCOMM_DEVVA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Retrieve the device-mapped virtual address for a host pointer that has
 * already been registered via HcommMemReg / MemMappingManager.
 *
 * Returns 0 on success, non-zero (HcclResult) on failure.
 * On success *dev_va is the device-side VA that corresponds to host_ptr.
 */
int hcomm_get_dev_va(int device_logic_id,
                     void *host_ptr,
                     uint64_t size,
                     void **dev_va);

#ifdef __cplusplus
}
#endif

#endif /* HCOMM_DEVVA_H */
