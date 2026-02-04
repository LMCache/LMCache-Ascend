# CacheBlend Implementation Guide

## Capability Matrix

The following table defines the support level of CacheBlend across various component combinations:

| vLLM Version | LMCache Version | Status | Stability Level | CacheBlend Support | Note |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **0.9.2** | **0.3.3 / 0.3.7** | 🟢 Stable | **Production Ready** | **Full** | - |
| **0.10.0** | **0.3.7** | 🔴 Unsupport | **N/A** | **None** | - |
| **0.11.0** | **0.3.7** | 🔴 Unsupport | **N/A** | **None** | - |
| **0.9.2** | **0.3.12** | 🔴 Conflict | **N/A** | **None** | Version mismatch; usage not advised. |
| **0.10.0** | **0.3.12** | 🟢 Stable | **Production Ready** | **Full** | - |
| **0.11.0** | **0.3.12** | 🟢 Stable | **Production Ready** | **Full** | - |

## 1. Critical Preparations

### 1.1 vLLM-Ascend Ad-Hoc Modifications

These temporary (ad-hoc) modifications are necessary for the cacheblend feature, based on instructions found here:https://github.com/LMCache/LMCache/blob/dev/examples/blend_kv_v1/README.md

Note: These patches are now automatically applied during the pip install process. No manual intervention is typically required. However, if you encounter issues when using CacheBlend, you can re-apply the patch using one of the methods below.

#### Option 1: Automatic Patching (Recommended)

You can apply these changes automatically without needing to reinstall vllm-ascend from source. Simply run the following command:

```
python /LMCache-Ascend/lmcache_ascend/integration/patch/apply_patch.py
```

#### Option 2: Manual Modification
If you prefer to update the code manually, please modify the following file:

**File Path**: `vllm-ascend/vllm-ascend/worker/worker_v1.py`
1. In the `_init_worker_distributed_environment` function: Comment out the line `ensure_kv_transfer_initialized(vllm_config)`
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