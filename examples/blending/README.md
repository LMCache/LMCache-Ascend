# CacheBlend Implementation Guide

## 1. Critical Preparations
### 1.1 Bug Fix & Patch Application
A known bug in **CacheBlend** (LMCache v0.3.7) causes low cache hit rates, with a fix scheduled for the v0.3.10 release. Apply this patch before proceeding:
```bash
cd /workspace/LMCache
git apply /workspace/LMCache-Ascend/lmcache_ascend/v1/blend/cacheblend_patch.diff
```

### 1.2 vLLM-Ascend Ad-Hoc Modifications
Temporary modifications are required to enable CacheBlend in vLLM-Ascend, following guidelines from the official LMCache documentation: <https://github.com/LMCache/LMCache/blob/dev/examples/blend_kv_v1/README.md>

**File Path**: `vllm-ascend/vllm-ascend/worker/worker_v1.py`
1. In the `init_worker_distributed_environment` function: Comment out the line `ensure_kv_transfer_initialized(vllm_config)`
2. At the end of the `load_model` function: Add the following Python snippet
```python
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME
        
VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
ensure_kv_transfer_initialized(self.vllm_config)
```

## 2. Configuration
### 2.1 Core Parameters
CacheBlend behavior is governed by the following core parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **enable_blending** | `bool` | `False` | Master switch for cache blending mode |
| **blend_special_str** | `str` | `" # # "` | Separator string for marking segment boundaries |
| **blend_recompute_ratios** | `Optional[list[float]]` | `None` | Ratios for recomputation decisions during blending |
| **blend_check_layers** | `list[int]` | `None` | Specific layer indices to validate during blending |
| **blend_min_tokens** | `int` | `256` | Minimum tokens required to activate cache blending |

### 2.2 Configuration Methods & Example
Configure CacheBlend via either of the following methods:
1. **YAML File** (recommended for persistent settings) Set the configuration file path using the environment variable: `LMCACHE_CONFIG_FILE="$config_file"`.
2. **Environment Variables** Define parameters directly in the environment, prefixed with `LMCACHE_` (e.g., `LMCACHE_ENABLE_BLENDING=true`).

#### YAML Configuration Example
```yaml
enable_blending: true
blend_special_str: " # # "
blend_min_tokens: 256
blend_check_layers: [1]
blend_recompute_ratios: 0.15
```


## 3. Current Limitations

### Feature Limitations
* **Prefix Cache Incompatibility:** CacheBlend and the **prefix cache cannot be enabled simultaneously**. Enabling one feature requires the other to be disabled.
* **Supported Architectures:** Currently limited to **Llama** and **Qwen3** model architectures.
* **Manual Segmentation Required:** Users must **manually insert** the defined separator string (`blend_special_str`) to correctly delineate segments within the prompt sequence.
    * *Example Prompt Structure:* `sys_prompt + blend_special_str + chunk1_prompt + blend_special_str + chunk2_prompt`
* **Offline Mode Token Requirement:** When processing in offline mode, the **BOS (Begin-of-Sequence) token must be manually removed** from all subsequent context segments (i.e., tokens following the initial `sys_prompt`).
```python
# Recommended
self.sep_tokens = self.tokenizer.encode(config.blend_special_str, add_special_tokens=False)

# (to be replaced)
self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
```


## 4. Usage & Validation

### 4.1 Basic Usage

`python script.py model_path 0.05` should yield no connector (ie `# #`) while `python script.py model_path 1.0` should.

The latter should yield the same result as `python script.py model_path 1.0 --no-blend`, although some precision issues may cause some difference.

### 4.2 Tested Models

```
Qwen/Qwen3-8B
mistralai/Ministral-8B-Instruct-2410
meta-llama/Meta-Llama-3.1-8B-Instruct
```