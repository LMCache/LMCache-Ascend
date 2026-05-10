---
name: upgrade-lmcache-ascend-version
description: |
  Upgrade LMCache-Ascend to be compatible with a new upstream LMCache version.
  This skill takes the current LMCache-Ascend monkey patches, re-derives them by
  comparing against the old LMCache version, and applies them to the new LMCache
  version's structure. Use whenever the user wants to upgrade the plugin for a
  new LMCache release, e.g. "upgrade to LMCache v0.5.0", "bump upstream version",
  "adapt to LMCache v0.6.0", "new LMCache version requires updates".
compatibility: |
  Requires git SSH access to git@github.com:LMCache/LMCache.git.
  Requires /tmp/lmcache_old and /tmp/lmcache_new for cloned sources.
  Requires pytest and the LMCache-Ascend dev environment.
---

# Upgrade LMCache-Ascend to a New Upstream Version

This skill re-derives all monkey patches by computing the diff between the
current LMCache-Ascend implementation and the old upstream LMCache source, then
applying those diffs to the new upstream version.

## Golden Rule: Always Check Cascading Effects

**Every upstream change can have cascading effects on the NPU implementation.
You MUST analyze and propagate changes, not just apply isolated diffs.**

Before making any change, ask:
1. **What upstream files/functions/classes changed?**
2. **What does this change call or depend on downstream?**
3. **Are there NPU-specific implementations that inherit from or wrap the changed code?**

Common cascading patterns:
- A factory function gains a parameter → all its callers (including NPU variants) must pass it
- A base class's `from_metadata()` gains a parameter → subclasses that override it must update
- A utility function/class is added to upstream `utils.py` → may need importing in NPU code
- A new connector type is added to upstream → only matters if it replaces existing NPU paths
- `_patch_config` adds config keys → existing config code in NPU may need updating

---

## Phase 1 — Read Current State

Read these files to understand the current versions:
- `lmcache_ascend/__init__.py` — find `LMCACHE_UPSTREAM_TAG` (e.g. `v0.4.2`)

**Extract target version from the user's prompt first.** If the prompt mentions a specific
version (e.g. "upgrade to LMCache v0.4.3" or "v0.4.3"), use that as the target.
Only ask the user via `AskUserQuestion` if no version is specified in the prompt.

---

## Phase 2 — Clone Old and New LMCache Sources

Clone the LMCache repo at both the old tag (current `LMCACHE_UPSTREAM_TAG`) and the new tag:

```bash
git clone --branch {old_tag} --depth 1 git@github.com:LMCache/LMCache.git /tmp/lmcache_old
git clone --branch {new_tag} --depth 1 git@github.com:LMCache/LMCache.git /tmp/lmcache_new
```

Example: if old tag is `v0.4.2` and new tag is `v0.5.0`:
```bash
git clone --branch v0.4.2 --depth 1 git@github.com:LMCache/LMCache.git /tmp/lmcache_old
git clone --branch v0.5.0 --depth 1 git@github.com:LMCache/LMCache.git /tmp/lmcache_new
```

Use `Bash` to run these commands. The `--depth 1` flag minimizes clone time.

---

## Phase 3 — Compute Diffs for Each Monkey Patch

For each of the following patches, read the three-way comparison and compute the diff:

### Patch Table (Dynamic)

The patch functions are **not hardcoded** — they are derived at runtime by analyzing the current `lmcache_ascend/__init__.py`.

At the start of Phase 3, **read the current `__init__.py` and automatically extract** all `_patch_*` function names and their corresponding source paths from the patch execution section. Use this to build a dynamic patch table for the current upgrade.

To extract the patch table:
1. Read `lmcache_ascend/__init__.py`
2. Find all `_patch_*` function definitions
3. Find all `_patch_*` calls in the patch execution section
4. Map each patch function to its target file based on the patterns in the existing code

### How to derive the patch table dynamically

The current `__init__.py` contains patch functions. Each patch function typically contains:
- A comment indicating the source file it patches (e.g., `# Patched from lmcache/v1/storage_backend/__init__.py`)
- The actual patched source path embedded in imports or references

Use these indicators to build the mapping rather than hardcoding it.

For each patchable entry:

1. Read `/tmp/lmcache_old/lmcache/{path}` — the original lmcache source at old tag
2. Read `lmcache_ascend/{path}` — the current lmcache_ascend implementation (current directory)
3. Compute the **diff**: what lines were changed/added/removed in lmcache_ascend vs the old upstream
4. Read `/tmp/lmcache_new/lmcache/{path}` — the new upstream source
5. **CRITICAL: Analyze cascading effects** (see below)

### How to compute the diff

Use Python's `difflib.unified_diff`. For each file pair:

```python
import difflib

def compute_diff(old_text, new_text, fromfile, tofile):
    """Return a list of diff line tuples (tag, old_line, new_line)."""
    diff_lines = []
    for line in difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=fromfile, tofile=tofile, lineterm=''
    ):
        diff_lines.append(line)
    return diff_lines
```

For each patch, you need to understand:
- Which **specific functions/classes** in the file were modified
- What the **exact changes** were (added lines, removed lines, modified lines)

### Naming conventions (apply during diff application)

When re-applying diffs to the new version:

- If the patch only adds NPU-specific behavior to an otherwise identical function: **keep the same name**
- If the patch replaces a GPU class with an NPU one: rename `GPUConnector` → `NPUConnector`, `CudaIPCWrapper` → `AscendIPCWrapper`
- If patching a class method: use `ClassName_method_name` naming (e.g. `GPUMemoryAllocator__init__`)
- Add `# LMC-A: <reason>` comment on every line that differs from the new upstream source. Example:
  ```python
  # LMC-A: Return NPU device instead of CUDA for Ascend compatibility
  return (torch.npu, "npu")
  ```

---


## Preserve Upstream Exactly — Only Diff Lines May Change

**Principle: Apart from parts that LMCache-Ascend explicitly needs to modify, all code — including comments, type annotations, and imports — must remain identical to upstream LMCache.**

When applying diffs to patched files, LMCache-Ascend should only diverge from upstream in the **exact lines** that constitute the diff. Everything else (whitespace, comments, docstrings, type hints, imports, line ordering) stays the same.

Concretely:

| Category | Rule |
|----------|------|
| **Type annotations** | If upstream adds a type to a parameter (e.g., `skip_backends: Optional[AbstractSet[str]]`), the LMCache-Ascend version **must also have** that type annotation. Only deviate when NPU-specific code genuinely requires different types. |
| **Imports** | If upstream adds a new import (e.g., `from typing import AbstractSet`), add it to LMCache-Ascend's imports. Do not remove or reorder upstream imports unless the diff explicitly requires it. |
| **Comments/docstrings** | Do not add, remove, or modify upstream comments or docstrings outside of lines that are already part of the diff. The `# LMC-A: <reason>` comment only goes on lines that actually differ from upstream. |
| **Line ordering** | Keep the same line order as upstream. If the diff inserts 3 lines at line 50, insert them at the same location — do not reorder surrounding code. |
| **Whitespace** | Preserve upstream indentation and spacing exactly. Only change whitespace on lines that are part of the diff. |

### Why this matters

If LMCache-Ascend modifies code outside the diff, it becomes impossible to cleanly see what NPU-specific changes were actually made. A future upgrade would compute a larger-than-necessary diff, making it hard to distinguish intentional NPU changes from incidental formatting/style changes. Upstream may also add new code (e.g., new parameters with type annotations) that LMCache-Ascend should inherit automatically — not override accidentally.

### Example of correct patching

```python
# upstream has:
def CreateStorageBackends(
    config: LMCacheEngineConfig,
    skip_backends: Optional[AbstractSet[str]] = None,  # type added in upstream
):
    ...

# LMCache-Ascend must also have (only diff lines are marked):
def CreateStorageBackends(
    config: LMCacheEngineConfig,
    skip_backends: Optional[AbstractSet[str]] = None,  # same as upstream
):
    # LMC-A: Use NPU device instead of CUDA
    device, device_type = (torch.npu, "npu")  # only this line differs
```

---

## Phase 4 — Apply Diffs to New LMCache

For each patch, apply the computed changes to the new upstream source:

### 4a. File still exists in new_lmcache at the same path

Apply the same changes (identified from comparing old_lmcache vs lmcache_ascend)
to the new_lmcache version of the file.

Example: if `CreateStorageBackends` in old lmcache used `is_cuda_worker` and
the ascend version changed it to `is_npu_worker`, apply the same change to
the new lmcache's `CreateStorageBackends`.

### 4b. File was renamed or moved in new_lmcache

Search for the modified function/class in the new_lmcache to find its new location.
If found at a new path, apply the diff there. If the function/class was removed,
the patch is no longer needed — note it for removal from `__init__.py`.

### 4c. _patch_config (special case)

This patch adds entries to `lmcache.v1.config._CONFIG_DEFINITIONS`. To handle:
1. Read old lmcache's `_CONFIG_DEFINITIONS` (from `/tmp/lmcache_old/lmcache/v1/config.py`)
2. Read new lmcache's `_CONFIG_DEFINITIONS` (from `/tmp/lmcache_new/lmcache/v1/config.py`)
3. Read current `__init__.py` to see what config keys were added
4. Add the new keys to the `_patch_config()` function in the new `__init__.py`

### 4d. c_ops handling

DO NOT attempt to diff/rebuild the `.so` file. Instead:
1. Compare old and new lmcache's `c_ops` Python stub files (`.pyi` or thin Python wrappers)
2. If the new lmcache added new pybind functions that are not in the ascend `c_ops.so`:
   - Read `lmcache_ascend/c_ops.pyi`
   - Add empty stub signatures for any new functions
   - Add `# LMC-A: <reason>` comments

### 4e. Cascading Effects Checklist (for every changed patch)

After reading the old vs new upstream source for each patch, **before writing any code**,
ask and answer these questions for the current patch:

#### Q1: What downstream callers exist?
Check all files in the upstream repo that import or call the changed function/class:
```bash
grep -r "from lmcache.v1.gpu_connector" /tmp/lmcache_new/lmcache/
grep -r "CreateGPUConnector" /tmp/lmcache_new/lmcache/
grep -r "from_metadata" /tmp/lmcache_new/lmcache/v1/gpu_connector/
```
If the changed function is a factory (`Create*`) or base class (`from_metadata`),
it will have cascading effects.

#### Q2: Does this patch's NPU target file have subclasses or wrappers?
Check if any NPU code **inherits from** or **wraps** the changed class:
```bash
grep -r "VLLMPagedMemGPUConnectorV2\|VLLMBufferLayerwiseGPUConnector\|VLLMPagedMemLayerwiseGPUConnector" lmcache_ascend/
grep -r "class.*Connector.*GPUConnector" lmcache_ascend/
```

#### Q3: Does the NPU subclass override the changed method?
If the NPU code inherits from a GPU class that changes, check whether the NPU version
**overrides** the method. Use `grep` to find `def method_name` in the NPU file.

| Situation | Action |
|----------|--------|
| NPU overrides method, parent signature changed | Add new params to NPU override, pass to `super()` |
| NPU inherits method without override | Usually works — but verify the parent signature change is backward-compatible |
| NPU is a factory function calling changed class | Update factory to pass new params |
| NPU uses changed utility class/function | Import it if needed |

#### Q4: Are new types/classes imported upstream that need importing in NPU?
When upstream adds a new import (e.g., `LayoutHints` TypedDict from `gpu_connectors.utils`),
check if the NPU `npu_connectors.py` also uses those types. If so, add the import.

#### Q5: Did `_patch_config` add new upstream config keys?
If upstream `config.py` added new `CONFIG_DEFINITIONS` keys, check whether the NPU's
patch config needs updating.

#### Concrete cascade examples from past upgrades:

**v0.4.2→v0.4.3 — `gpu_connector` cascade:**
- Upstream `CreateGPUConnector` gained `layout_hints: LayoutHints | None = None`
- Upstream `VLLMPagedMemGPUConnectorV2.from_metadata()` gained `layout_hints` param
- → `VLLMPagedMemNPUConnectorV2.from_metadata()` **overrides** the parent → must add `layout_hints`
- → `VLLMBufferLayerwiseNPUConnector` and `VLLMPagedMemLayerwiseNPUConnector` **inherit**
  without override → work automatically
- → `CreateNPUConnector` was **not passing** `layout_hints` → must fix

**v0.4.x — `kv_layer_groups` cascade:**
- Upstream `build_kv_layer_groups` added tuple/list KV cache format support
- → NPU `build_kv_layer_groups` had its own tuple handling for Ascend format
- → NPU version must support **both** the new upstream format **and** the old Ascend format

---

## Phase 5 — Write New Files to lmcache_ascend/

For each generated new implementation, use the `Write` tool to write it to the
appropriate path under `lmcache_ascend/` (current directory).

Path mapping for writes:
- Diff output for `lmcache/v1/foo.py` → write to `lmcache_ascend/v1/foo.py`
- Diff output for `lmcache/integration/vllm/foo.py` → write to `lmcache_ascend/integration/vllm/foo.py`
- Diff output for `lmcache/integration/sglang/foo.py` → write to `lmcache_ascend/integration/sglang/foo.py`

Use the exact same directory structure as lmcache_ascend already has. Create
directories as needed with the `Bash` tool.

---

## Phase 6 — Update __init__.py

Read the current `lmcache_ascend/__init__.py` and update:

### 6a. Update LMCACHE_UPSTREAM_TAG
```python
LMCACHE_UPSTREAM_TAG = "{new_version}"
```

### 6b. Remove obsolete patch functions

If a patch target no longer exists in the new lmcache (function deleted, class renamed):
1. Remove the corresponding `_patch_*` function body
2. Remove its import statement at the top of `__init__.py`
3. Remove its call from the patch execution section (lines ~463-505)

### 6c. Verify patch ordering

Ensure the execution order in `__init__.py` is preserved. Required ordering:
1. `_patch_config` (always first)
2. `_patch_torch_capability` (pytorch framework only)
3. `_patch_ops` (always)
4. For vllm runtime: `_patch_get_vllm_torch_dev`, then `_patch_gpu_connector`
5. `_patch_hash_token` (always)
6. For pytorch framework: `_patch_storage_backend_init`, `_patch_transfer_channel`, `_patch_cacheblend`, `_patch_multi_process`, `_patch_lookup_client`, `_patch_rpc_utils`
7. `_patch_kv_layer_group` (always)
8. For sglang runtime: `_patch_sgl`
9. For vllm runtime + pytorch: `_patch_sys_detection`
10. For vllm runtime: `_patch_vllm_v1_adapter`, then `_patch_cache_engine`

Key dependency: For vllm, `gpu_connector` must be patched before `storage_backend_init`
because `CreateStorageBackends` (patched by `storage_backend_init`) calls `CreateNPUConnector`
internally at runtime. The patch order ensures the NPU factory is registered before
`CreateStorageBackends` can use it.

---

## Phase 7 — Update CI and README

### 7a. Update CI workflow

Read `.github/workflows/build-and-test.yml` and update ALL occurrences of:
```
pip install lmcache=={old_version}
```
to:
```
pip install lmcache=={new_version_without_v}
```

There are typically 2 occurrences (910B job and 910C job).

### 7b. Update Dockerfiles

Update all Dockerfiles in `docker/` that reference the old upstream LMCache version:

Find and replace in ALL of these files:
- `docker/Dockerfile.a2.openEuler` - line 30: `pip install lmcache=={old}` -> `pip install lmcache=={new}`
- `docker/Dockerfile.a3.openEuler` - line 30: `pip install lmcache=={old}` -> `pip install lmcache=={new}`
- `docker/Dockerfile.a3` - line 30: `pip install lmcache=={old}` -> `pip install lmcache=={new}`
- `docker/Dockerfile.310p.openEuler` - line 30: `pip install lmcache=={old}` -> `pip install lmcache=={new}`
- `docker/mindspore/Dockerfile.310p.openEuler` - line 29: `pip install lmcache=={old}` -> `pip install lmcache=={new}`
- `docker/mindspore/Dockerfile.a2.openEuler` - line 29: `pip install lmcache=={old}` -> `pip install lmcache=={new}`

Use the `Edit` tool with `replace_all: true` on each file for the `pip install lmcache==` or `LMCACHE_TAG=` lines.

### 7c. Update README compatibility matrix

Read `README.md`. The compatibility table has columns: LMCache-Ascend version,
Upstream LMCache, vLLM/SGLang version, CANN version. Update:
- The row for the current branch (`main`) to show the new upstream LMCache version
- If adding a new verified row, mirror the format of existing rows

### 7d. Bump version in README and Dockerfiles (version-only refresh)

This step updates version numbers across documentation and Docker files when
LMCache-Ascend has a new release that only bumps the upstream LMCache version
(no source code changes to the NPU plugin). It is **not** the main upgrade
skill — use this when the upstream LMCache version changed but no monkey-patch
re-derivation is needed (e.g. releasing v0.4.3 after v0.4.2, where lmcache
version also went from 0.4.2 to 0.4.3).

#### Step 1 — Update README Compatibility Matrix

For each row in the compatibility matrix (SGLang and MindSpore sections),
update only the **first two columns** (LMCache-Ascend version and Upstream
LMCache) from the old version to the new version. Leave all other columns
(vLLM/SGLang version, PyTorch/Torch-NPU/MindSpore, Status) **unchanged**.

Example — before:
```
| **v0.4.2** | **v0.4.2** | **0.5.8** | **2.8.0.post2.dev20251113** | ✅ **Verified (Recommended)** |
```
After (only first two columns changed):
```
| **v0.4.3** | **v0.4.3** | **0.5.8** | **2.8.0.post2.dev20251113** | ✅ **Verified (Recommended)** |
```

#### Step 2 — Update README Installation Sections

Find and replace all occurrences in `README.md`:
- `git clone --recurse-submodules -b v{old}` → `git clone --recurse-submodules -b v{new}`
- `pip install lmcache=={old}` (all variants) → `pip install lmcache=={new}`
- Docker image tags `lmcache-ascend:v{old}-*` → `lmcache-ascend:v{new}-*`
- `docker build ... -t lmcache-ascend:v{old}-` → `docker build ... -t lmcache-ascend:v{new}-`

Use `Edit` tool with `replace_all: true` for each pattern.

#### Step 3 — Update all Dockerfiles

Find and replace in all Dockerfiles under `docker/`:
- `pip install lmcache=={old}` → `pip install lmcache=={new}`

Use `Edit` tool with `replace_all: false` (target the specific line in each file).

Files to update:
- `docker/Dockerfile.310p.openEuler`
- `docker/Dockerfile.a2.openEuler`
- `docker/Dockerfile.a3`
- `docker/Dockerfile.a3.openEuler`
- `docker/mindspore/Dockerfile.310p.openEuler`
- `docker/mindspore/Dockerfile.a2.openEuler`

---

## Phase 8 — Run Tests

Run the test suite to verify the upgraded code:

```bash
cd /mnt/sdb/jjy/LMCache-Ascend
python3 -m pip install -v --no-build-isolation -e . 2>&1 | tail -5
python3 -m pytest -v tests/v1 -x --tb=short 2>&1 | tail -30
```

### If tests fail — Auto-fix and retry (up to 3 iterations)

For each test failure:
1. Read the failing test file to understand what it expects
2. Identify which patch is likely responsible for the failure
3. Read the relevant new lmcache source to understand what changed
4. Update the diff / generated file for that patch
5. Re-write the affected file
6. Re-run tests

If after 3 retry cycles tests are still failing, stop and report:
- Which test is failing
- Which patch is responsible
- What specifically changed in the new lmcache that broke it
- Ask the user how they want to proceed

---

## Phase 9 — Cleanup

After all patches are applied and tests pass, remove the cloned sources:

```bash
rm -rf /tmp/lmcache_old /tmp/lmcache_new
```

---

## Summary of Generated/Updated Files

The skill updates files based on the dynamically derived patch table. After extracting the patch functions from `__init__.py`, only the files that actually need patching are updated. All file paths are relative to the current directory (lmcache_ascend/).

Typical files that may be updated include:
- `__init__.py` — `LMCACHE_UPSTREAM_TAG`, patch functions
- `v1/storage_backend/__init__.py`
- `v1/transfer_channel/__init__.py`
- `v1/blend/utils.py`
- `v1/multiprocess/custom_types.py`
- `v1/kv_layer_groups.py`
- `v1/gpu_connector/__init__.py`
- `v1/rpc_utils.py`
- `v1/tokens_hash.py`
- `v1/token_database.py`
- `v1/lookup_client/lmcache_lookup_client.py`
- `v1/system_detection.py`
- `v1/memory_management.py`
- `integration/vllm/utils.py`
- `integration/vllm/vllm_v1_adapter.py`
- `integration/sglang/sglang_adapter.py`
- CI and documentation files (`.github/workflows/build-and-test.yml`, `README.md`, `docker/Dockerfile.*`)
