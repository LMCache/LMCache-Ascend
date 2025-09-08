import os
import argparse
from transformers import AutoTokenizer
import contextlib
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm import LLM, SamplingParams
from dataclasses import asdict

@contextlib.contextmanager
def build_llm_with_lmcache(model: str, max_model_len: int=32000, blend: bool=True):
    lmcache_connector = "LMCacheAscendConnectorV1Dynamic"
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
        kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )

    if blend:
        llm_args = EngineArgs(
            model=model,
            kv_transfer_config=ktc,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.6,
            enable_prefix_caching=False,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            trust_remote_code=True,
        )
    else:
        llm_args = EngineArgs(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.6,
            enable_prefix_caching=False,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            trust_remote_code=True,
        )

    llm = LLM(**asdict(llm_args))
    yield llm

def setup_environment_variables(recompute_ratio):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["VLLM_ENFORCE_EAGER"] = "True"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIO"] = str(recompute_ratio)

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SELECTION_W"] = "-1"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "# #"
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["VLLM_USE_V1"] = "1"

    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"

    # Set the maximum size of the local CPU size to 5GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

def parse_args():
    parser = argparse.ArgumentParser(description="Example script for blending")
    parser.add_argument("model", help="path to the model")    
    parser.add_argument("recompute_ratio", help="recompute ratio for blending")    
    parser.add_argument("--no-blend", action='store_true', help="")    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = args.model
    recompute_ratio = args.recompute_ratio
    setup_environment_variables(recompute_ratio)
    tokenizer = AutoTokenizer.from_pretrained(model)
    chunk1_prompt = tokenizer.encode("By definition x=2;", add_special_tokens=False)
    chunk2_prompt = tokenizer.encode("By definition y=7;", add_special_tokens=False)
    chunk3_prompt = tokenizer.encode("Then x+y=", add_special_tokens=False)
    blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"), add_special_tokens=False)
    prompt = (
          chunk1_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk3_prompt
    )
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
    with build_llm_with_lmcache(model, blend=not args.no_blend) as llm:
        llm.generate({"prompt_token_ids": chunk1_prompt + blend_special_str}, sampling_params=sampling_params)
        llm.generate({"prompt_token_ids": chunk2_prompt + blend_special_str}, sampling_params=sampling_params)
        llm.generate({"prompt_token_ids": chunk3_prompt}, sampling_params=sampling_params)
        outputs1 = llm.generate({"prompt_token_ids": prompt}, sampling_params=sampling_params)
        print(f"Output: {outputs1[0].outputs[0].text}")