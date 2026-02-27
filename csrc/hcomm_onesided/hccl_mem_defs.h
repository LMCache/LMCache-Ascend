// SPDX-License-Identifier: Apache-2.0
// Shim header: hccl_one_sided_services.h includes "hccl_mem_defs.h" but it is
// not shipped with the installed CANN package but in the hccl directory. 
// We provide the minimal types that the one-sided services header needs.

#ifndef HCCL_MEM_DEFS_H
#define HCCL_MEM_DEFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HCCL_MEM_TYPE_DEVICE = 0,
    HCCL_MEM_TYPE_HOST = 1,
    HCCL_MEM_TYPE_NUM
} HcclMemType;

typedef struct {
    HcclMemType type;
    void *addr;
    uint64_t size;
} HcclMem;

#ifdef __cplusplus
}
#endif
#endif // HCCL_MEM_DEFS_H
