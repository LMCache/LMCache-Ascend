# Concurrent CPU-restore correctness evaluation

## Status

The evaluation client and its host regressions are provided here. The new
real-device synchronous/asynchronous answer comparison has **not yet been run**.
Populate the result record below after running both configurations. Historical
latency measurements and host protocol tests are separate evidence.

## Workload and acceptance criteria

The client generates deterministic long documents containing a unique random
16-hex-digit passkey. Each measured question asks for that document's passkey;
the answer is present only inside the context, not in the question. Needle
positions are approximately 10%, 50% and 90% of the document. The server's
tokenizer fits each measured prompt to 32,768..33,024 tokens by default.

For each concurrency (1, 2 and 4), depth and repeat:

1. Warm each distinct context using a different question, retaining its CPU KV
   and GDN checkpoints. Wait for the service to settle.
2. Pause with `mode=wait&clear_cache=false`, then call
   `/reset_prefix_cache?reset_running_requests=false&reset_external=false`.
   This clears only native prefix reuse; it does not clear the CPU cache.
3. Resume and send a barrier-started concurrent wave of passkey questions.
4. Require every response to report positive cached tokens; the external-hit
   counter delta must equal their sum. Native-hit and preemption deltas must be
   zero. Each needle must lie inside that request's restored prefix, rather than
   a recomputed tail. Missing counters, unsupported control APIs or a rejected reset stop
   the run instead of silently accepting a different path.
5. Score the returned passkeys. Preserve full answers, prompt hashes, token
   usage and counter snapshots in JSON, including failed path-verification waves.

Default coverage is nine waves / 21 measured requests per configuration. This
is a focused long-context regression screen, not a broad model-quality benchmark.
Increase `--repeats` for stronger coverage. A second 65,536-token run can exercise
longer restore boundaries, subject to the configured CPU/NPU capacity.

The comparison requires identical case IDs, prompt hashes and expected answers
between synchronous and asynchronous runs, and at least one concurrent wave.
All expected answers must be correct in both configurations; any wrong answer
or unverified CPU-restore wave is a failure to investigate.

## Deployment preparation

Use a **dedicated test server** with the same reviewed plugin checkout for both
runs, paired with LMCache 0.4.5. Keep hardware, model/tokenizer revisions, BF16,
TP=2, chunk size, hybrid-cache layout, MTP settings and graph mode fixed. Record
the exact plugin commit, loaded module paths, startup commands and versions.
Installing this checkout as an editable package must not leave an older overlay
ahead of it on `PYTHONPATH`.

Use the existing validated Qwen serving command and change the following settings:

| Setting | Synchronous baseline | Asynchronous configuration |
| --- | --- | --- |
| LMCache config | `examples/gdn/sync.yaml` | `examples/gdn/async.yaml` |
| `LMCACHE_CONFIG_FILE` | Absolute path to sync.yaml | Absolute path to async.yaml |
| vLLM scheduling | `--no-async-scheduling` | `--async-scheduling` |
| Native prefix cache | Enabled | Enabled |
| Hybrid state mode | `--mamba-cache-mode align` | Same |
| Usage details | `--enable-prompt-tokens-details` | Same |
| Development controls | `VLLM_SERVER_DEV_MODE=1` | Same |

The supplied YAML sets explicit GDN flags, including false for the baseline, so
legacy environment opt-ins do not override it. Restart the test service between
configurations. The client never starts, stops or reconfigures a model process.
Keep development controls on the dedicated local/test endpoint.

## Run on the device host

Run from the repository root, using the serving environment's Python. Replace
the example model name if the service exposes a different name. The client is
stdlib-only; it uses the server's `/tokenize` endpoint.

After starting the synchronous baseline:

```bash
python examples/gdn/validate_async_correctness.py run --mode sync --model qwen3-30b-a3b-thinking --config examples/gdn/sync.yaml --suite-id gdn-review-20261029 --context-tokens 32768 --concurrencies 1 2 4 --depths 0.1 0.5 0.9 --output sync-correctness.json --dedicated-test-server
```

After stopping that service and starting the async configuration on the same
dedicated endpoint:

```bash
python examples/gdn/validate_async_correctness.py run --mode async --model qwen3-30b-a3b-thinking --config examples/gdn/async.yaml --suite-id gdn-review-20261029 --context-tokens 32768 --concurrencies 1 2 4 --depths 0.1 0.5 0.9 --output async-correctness.json --dedicated-test-server
python examples/gdn/validate_async_correctness.py compare sync-correctness.json async-correctness.json
```

The default endpoint is `http://127.0.0.1:8010`; use `--base-url` to change it.
Optional authentication uses `OPENAI_API_KEY` without recording its value.
Output files must not already exist. Keep the same suite ID and generation
settings for both modes. The config hash records the supplied file, not proof
that the server loaded it; retain the server's startup/configuration evidence.

If CPU hit verification fails, inspect the JSON and both TP worker logs before
retrying. Confirm store completion, checkpoint availability and sufficient CPU
capacity. Do not bypass the path checks or substitute native NPU hits.

## Result record (pending device execution)

| Item | Synchronous | Asynchronous |
| --- | --- | --- |
| Plugin commit / actual module path | Pending | Pending |
| Model/tokenizer, Ascend toolkit, vLLM and LMCache versions | Pending | Pending |
| Startup command and config hash | Pending | Pending |
| Verified waves / requests | Pending | Pending |
| Passkey accuracy at each concurrency/depth | Pending | Pending |
| External hits, native hits and preemptions | Pending | Pending |
| Both TP ranks' receive/fence evidence | Pending | Pending |
| JSON results and server logs | Pending | Pending |

Host regressions for the evaluation client's generation, path checks, reset
behavior and comparison can run without an NPU:

```bash
python -m unittest discover -s tests/host -p test_gdn_correctness_host.py -v
```
