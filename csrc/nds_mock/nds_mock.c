/*
 * SPDX-License-Identifier: Apache-2.0
 * Development mock for libdpc_nds.so (MindIO five-symbol contract).
 * Uses host pread/pwrite + aclrtMemcpy — not production NDS direct DMA.
 */
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "acl/acl.h"

typedef void *NdsFileHandle_t;

typedef enum NdsFileOpError {
  NDS_FILE_SUCCESS = 0,
  NDS_FILE_DRIVER_ERROR = -1,
} NdsFileOpError;

typedef struct NdsFileError_s {
  enum NdsFileOpError err;
} NdsFileError_t;

typedef struct NdsFileDescr_s {
  int fd;
} NdsFileDescr_t;

typedef struct NdsFileHandleInner {
  int fd;
} NdsFileHandleInner;

static int g_driver_open = 0;
static int g_acl_ready = 0;

static int ensure_acl(void) {
  if (g_acl_ready) {
    return 0;
  }
  if (aclInit(NULL) != ACL_SUCCESS) {
    return -1;
  }
  if (aclrtSetDevice(0) != ACL_SUCCESS) {
    return -1;
  }
  g_acl_ready = 1;
  return 0;
}

NdsFileError_t NdsFileDriverOpen(void) {
  NdsFileError_t out = {NDS_FILE_SUCCESS};
  if (ensure_acl() != 0) {
    out.err = NDS_FILE_DRIVER_ERROR;
  } else {
    g_driver_open = 1;
  }
  return out;
}

NdsFileError_t NdsFileDriverClose(void) {
  NdsFileError_t out = {NDS_FILE_SUCCESS};
  g_driver_open = 0;
  return out;
}

NdsFileError_t NdsFileHandleRegister(NdsFileHandle_t *fh, NdsFileDescr_t *descr) {
  NdsFileError_t out = {NDS_FILE_SUCCESS};
  if (!g_driver_open || fh == NULL || descr == NULL) {
    out.err = NDS_FILE_DRIVER_ERROR;
    return out;
  }
  NdsFileHandleInner *inner = calloc(1, sizeof(NdsFileHandleInner));
  if (inner == NULL) {
    out.err = NDS_FILE_DRIVER_ERROR;
    return out;
  }
  inner->fd = descr->fd;
  *fh = inner;
  return out;
}

void NdsFileHandleDeregister(NdsFileHandle_t fh) {
  if (fh != NULL) {
    free(fh);
  }
}

static ssize_t copy_to_device(void *dev_ptr, const void *host, size_t size) {
  aclError st =
      aclrtMemcpy(dev_ptr, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return (st == ACL_SUCCESS) ? (ssize_t)size : -1;
}

static ssize_t copy_from_device(void *host, const void *dev_ptr, size_t size) {
  aclError st =
      aclrtMemcpy(host, size, dev_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return (st == ACL_SUCCESS) ? (ssize_t)size : -1;
}

ssize_t NdsFileRead(NdsFileHandle_t fh, void *ptr_base, size_t size,
                    off_t file_offset, off_t ptr_offset) {
  NdsFileHandleInner *inner = (NdsFileHandleInner *)fh;
  if (inner == NULL || ptr_base == NULL || size == 0) {
    return -1;
  }
  void *host = NULL;
  if (posix_memalign(&host, 4096, size) != 0) {
    return -1;
  }
  ssize_t n = pread(inner->fd, host, size, file_offset);
  if (n != (ssize_t)size) {
    free(host);
    return -1;
  }
  char *dev = (char *)ptr_base + ptr_offset;
  ssize_t rc = copy_to_device(dev, host, size);
  free(host);
  return rc;
}

ssize_t NdsFileWrite(NdsFileHandle_t fh, void *ptr_base, size_t size,
                     off_t file_offset, off_t ptr_offset) {
  NdsFileHandleInner *inner = (NdsFileHandleInner *)fh;
  if (inner == NULL || ptr_base == NULL || size == 0) {
    return -1;
  }
  void *host = NULL;
  if (posix_memalign(&host, 4096, size) != 0) {
    return -1;
  }
  char *dev = (char *)ptr_base + ptr_offset;
  if (copy_from_device(host, dev, size) != (ssize_t)size) {
    free(host);
    return -1;
  }
  ssize_t n = pwrite(inner->fd, host, size, file_offset);
  free(host);
  return (n == (ssize_t)size) ? (ssize_t)size : -1;
}
