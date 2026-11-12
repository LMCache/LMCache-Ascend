# SPDX-License-Identifier: Apache-2.0
"""Compile the real allocator against controlled ACL calls, without CANN."""

# Standard
from pathlib import Path
import os
import shutil
import subprocess

# Third Party
import pytest


def test_pure_host_allocation_and_native_error_paths(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    include = tmp_path / "acl"
    include.mkdir()
    (include / "acl.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        "using aclError = int; constexpr int ACL_SUCCESS = 0;\n"
        "enum class aclrtLastErrLevel { ACL_RT_THREAD_LEVEL };\n"
        "int aclrtMallocHost(void **, size_t); int aclrtFreeHost(void *);\n"
        "const char *aclrtGetSocName(); int aclrtGetLastError(aclrtLastErrLevel);\n"
    )
    (tmp_path / "numaif.h").write_text(
        "#pragma once\n#define MPOL_BIND 1\n#define MPOL_MF_MOVE 2\n"
        "#define MPOL_MF_STRICT 4\n"
        "inline int mbind(void *, unsigned long, int, const unsigned long *, "
        "unsigned long, unsigned) { return 0; }\n"
    )
    source = tmp_path / "check.cpp"
    source.write_text(
        r"""
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "mem_alloc.h"
#include "acl/acl.h"
int allocations = 0, frees = 0, registrations = 0, unregistrations = 0;
int alloc_error = 0, free_error = 0;
int aclrtMallocHost(void **ptr, size_t size) {
  ++allocations;
  if (alloc_error) return alloc_error;
  *ptr = std::malloc(size);
  return 0;
}
int aclrtFreeHost(void *ptr) {
  ++frees;
  if (free_error) return free_error;
  std::free(ptr); return 0;
}
const char *aclrtGetSocName() { return "Ascend910"; }
int aclrtGetLastError(aclrtLastErrLevel) { return 0; }
void *register_ptr(void *ptr, size_t) { ++registrations; return ptr; }
int unregister_ptr(void *) { ++unregistrations; return 0; }
int main() {
  try { alloc_host_ptr(0); assert(false); }
  catch (const std::invalid_argument &) {}
  assert(allocations == 0);
  uintptr_t ptr = alloc_host_ptr(64);
  assert(ptr != 0);
  reinterpret_cast<unsigned char *>(ptr)[63] = 19;
  assert(reinterpret_cast<unsigned char *>(ptr)[63] == 19);
  free_error = 73;
  try { free_host_ptr(ptr); assert(false); }
  catch (const std::runtime_error &e) {
    std::string msg(e.what());
    assert(msg.find("aclrtFreeHost") != std::string::npos);
    assert(msg.find("73") != std::string::npos);
    assert(msg.find(std::to_string(ptr)) != std::string::npos);
  }
  free_error = 0; free_host_ptr(ptr);
  alloc_error = 91;
  try { alloc_host_ptr(127); assert(false); }
  catch (const std::runtime_error &e) {
    std::string msg(e.what());
    assert(msg.find("aclrtMallocHost") != std::string::npos);
    assert(msg.find("91") != std::string::npos);
    assert(msg.find("127") != std::string::npos);
  }
  assert(registrations == 0 && unregistrations == 0);
  alloc_error = 0;
  ptr = alloc_pinned_ptr(64, 0); free_pinned_ptr(ptr);
  assert(registrations == 1 && unregistrations == 1);
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "check"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(tmp_path),
            "-I",
            str(root / "csrc"),
            str(source),
            str(root / "csrc/mem_alloc.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True)
