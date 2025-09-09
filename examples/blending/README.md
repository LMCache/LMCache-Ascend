# ad-hoc modification to vllm-ascend

We took b5b7e0ecc765adfec4ec4efdd14bfc7cce715e23 of the official vllm-ascend repository (0.9.2) then made modifications from https://github.com/vllm-project/vllm-ascend/pull/2039 and added ad-hoc modifications based on https://github.com/LMCache/LMCache/blob/dev/examples/blend_kv_v1/README.md

The git diff file is at `examples/blending/since_b5b7e0ecc765adfec4ec4efdd14bfc7cce715e23.diff`.

# Usage

`python script.py model_path 0.05` should yield no connector (ie `# #`) while `python script.py model_path 1.0` should.

The latter should yield the same result as `python script.py model_path 1.0 --no-blend`, although some precision issues may cause some difference.

## Tested Models

```
Qwen/Qwen3-8B
mistralai/Ministral-8B-Instruct-2410
meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Notes

Current attention backend utilizes transformer style attention. This is because cacheblend requires custom causal attention masks.

Ideally in LMCache 0.3.3 `SegmentTokenDatabase`, `self.sep_tokens = self.tokenizer.encode(config.blend_special_str, add_special_tokens=False)` instead of `self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]` should be used to extract separation tokens.
For now, in the script for qwen tokenizers that do not have `BOS`, as in the example script, individual chunk caching should be done like `llm.generate({"prompt_token_ids": chunk1_prompt + blend_special_str}, sampling_params=sampling_params)` insetad of `llm.generate({"prompt_token_ids": chunk1_prompt}, sampling_params=sampling_params)`.