#pragma once
#include <cstddef>
#include <cstdint>

/*
 * These following APIs are called directly in LMCache,
 * therefore we assume the ptrs management are done by the python program for
 * now.
 */

// Pure ACL Host allocation, without registration or device mapping. The caller
// prepares the runtime and releases the original address after all users stop.
uintptr_t alloc_host_ptr(std::size_t size);

void free_host_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, std::size_t size);
