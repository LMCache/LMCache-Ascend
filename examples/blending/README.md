# ad-hoc modification to vllm-ascend

We took b5b7e0ecc765adfec4ec4efdd14bfc7cce715e23 of the official vllm-ascend repository (0.9.2) then made modifications from https://github.com/vllm-project/vllm-ascend/pull/2039 and added ad-hoc modifications based on https://github.com/LMCache/LMCache/blob/dev/examples/blend_kv_v1/README.md

The git diff file is at `examples/blending/since_b5b7e0ecc765adfec4ec4efdd14bfc7cce715e23.diff`.

# Usage

`python script.py model_path 0.05` should yield no connector (ie `# #`) while `python script.py model_path 1.0` should.

The latter should yield the same result as `python script.py model_path 1.0 --no-blend`, although some precision issues may cause some difference.