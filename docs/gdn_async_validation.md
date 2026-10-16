# GDN async extraction audit and retained device evidence

## Scope and version relationship

The reporter already tested a pinned local full-async deployment on two Ascend
NPUs with vLLM 0.23, LMCache core 0.4.5, TP=2, MTP=3, decode graphs, prefix
caching and a 32 GiB CPU cache per rank. Those results are retained below.
They are not discarded merely because the upstream implementation was extracted
into native modules rather than delivered as runtime method replacements.

The source snapshot reviewed for this update was PR #309 commit
`ae132fd69ad6470f95261ce9bb527ad11cc8b8ef`, targeting the GDN implementation
on `dsv4_support_045`. PR #307 is the separate native multimodal-bypass fix.
This feature does not include private BNSD kernels, fixed deployment paths,
status-file launch tooling, or the later availability/circuit-breaker overlay.

### Retained prototype delivery fingerprints

These are the retained local delivery source files used for the extraction
audit. Their presence is not proof that every benchmark process loaded these
exact bytes; the older run logs did not pin a Git commit for all loaded files.

| Delivery source | SHA256 |
| --- | --- |
| `full_async.py` | `5c78fdb0fc23930e670f8021a71ce40dfc40bc5e0a783f359c4bde5d787e3534` |
| `gdn_async_store.py` | `01f781f1328152231981b87051f1ea9a850428e8432facd80de8d662d3142d55` |
| `async_control.py` | `7d3f8678f89f13273b00cc916a47f6dd2a3c7245d82ea4a4a1ff53ba8926ce99` |

### Source mapping and intentional differences

| Prototype implementation | Upstream extraction | Audit result / host coverage |
| --- | --- | --- |
| `full_async.AsyncStateClient` | `v1/gdn_async_lookup.py` | Class AST is identical (ignoring source locations and formatting); thread ownership, pending polling and cancellation tested. |
| `Receive`, `acquire_load`, `make_receive` | `integration/vllm/gdn_async_receive.py` | Each symbol's AST is identical; framework/device interfaces still need exact-commit integration smoke. |
| `FrozenExecution`, `StateJob` | `v1/gdn_async_store.py` | Each class AST is identical; private cloning and request-config ownership tested. |
| Store method replacements and environment opt-ins | `GDNAsyncStoreMixin`, engine call sites, `gdn_async_config.py` | Native helper calls replace method patching. This update restores the measured launcher's environment aliases; explicit `extra_config` wins, including `false`. |
| Global factory/manager method replacements | `AscendGDNLookupFactory`, `AscendGDNManager` | Scoped opt-in and ordinary-path delegation are intentional changes; host tests exercise both modes without mutating shared config. |
| Transfer method wrappers | engine transfer lock and `perform_load` | Shared scratch remains serialized; the lock now remains held through a `finally` fence on partially submitted restore failures. Fence-before-release and quarantine tested with device doubles. |
| Completion/owner primitives | `v1/async_control.py` | This update fixes retrying close after full-queue timeout, atomic failure checks, and bounded unreported completions. These are not claimed AST-equivalent. |
| Status JSON and private availability policies | Not included | No file-system status protocol, automatic recomputation guarantee, or universal process-survival claim is added. |

Exact AST matches establish source preservation for those named symbols only,
not equivalence of the complete initialization/runtime path. New tests load the
actual extracted production logic with explicit dependency doubles; they are not
a second hand-written implementation that merely repeats the intended result.

## Existing NPU measurements (local prototype, not current upstream head)

The following values were transcribed from the reporter's original `SUMMARY`
records, with cache-path evidence checked against the corresponding `WAVE_PASS`
records. No new device run was performed to generate this table.

Common workload: input **65,280** tokens, output **256** tokens, **53,760** actual
cache-hit tokens per request (82.35%), **11,520** tokens recomputed. Fixed,
barrier-started waves; one warmup plus three measured waves at each concurrency,
giving 3/6/12 measured requests at concurrency 1/2/4. Cold preparation, drain and
cache reset are excluded from the measured request window. Prompts are synthetic
token sequences, not an answer-quality benchmark.

| Configuration | Actual path | Concurrency | TTFT P50 / P95 (s) | TPOT P50 (ms/token) | E2E P50 (s) | Window output (token/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Production sync reproduced | NPU prefix reuse | 1 | 3.803 / 3.808 | 24.11 | 9.945 | 25.60 |
| Production sync reproduced | CPU restore | 1 | 3.831 / 3.838 | 24.13 | 9.984 | 25.46 |
| Async LMCache prototype | NPU prefix reuse | 1 | 3.521 / 3.523 | 16.23 | 7.661 | 31.88 |
| Async LMCache prototype | CPU restore | 1 | 3.861 / 3.861 | 16.11 | 7.970 | 30.78 |
| Production sync reproduced | NPU prefix reuse | 2 | 7.052 / 7.494 | 28.16 | 14.099 | 35.75 |
| Production sync reproduced | CPU restore | 2 | 7.135 / 7.557 | 28.54 | 14.179 | 35.44 |
| Async LMCache prototype | NPU prefix reuse | 2 | 5.864 / 7.002 | 22.38 | 11.140 | 44.04 |
| Async LMCache prototype | CPU restore | 2 | 6.419 / 7.558 | 22.28 | 11.736 | 42.06 |
| Production sync reproduced | NPU prefix reuse | 4 | 12.024 / 15.237 | 45.81 | 22.798 | 43.23 |
| Production sync reproduced | CPU restore | 4 | 12.480 / 15.708 | 45.86 | 23.335 | 42.56 |
| Async LMCache prototype | NPU prefix reuse | 4 | 10.625 / 13.549 | 28.15 | 17.917 | 55.96 |
| Async LMCache prototype | CPU restore | 4 | 11.964 / 14.703 | 27.65 | 19.018 | 52.69 |

CPU-path proof in each measured async wave:

- NPU hit delta: 0; external hit delta: `53,760 * concurrency`.
- Every request reports 53,760 cached tokens.
- Each TP rank reports `enqueued == completed == concurrency`; checkpoint skip
  delta and preemption delta are zero in these waves.
- A separate 65,536-token restore smoke had NPU hit delta 0 and external hit
  delta 61,440 after clearing NPU cache only.

Limitations: the sync and async deployment configurations also differed in
private BNSD routing, so these are **combined deployment results, not an isolated
causal measurement of the async code in this PR**. They do not promise speedup
at every concurrency: single-request CPU TTFT is slightly worse in this sample.
P95 is a small-sample statistic; window throughput is not steady-state capacity.
Client TPOT is an approximation because MTP can emit multiple tokens per event.
Response completion/cache-hit evidence is not a numerical correctness proof.

## Validation added for the extracted source

Python 3.12 host run: 30 unique tests passed (9 multimodal plus 21 async).
Ruff checks/format and Python compilation also pass. The host CI workflow runs
the same stdlib test command on Python 3.10 and 3.12 without device dependencies;
its remote status must be checked separately, not inferred from the local run.

```bash
python -m unittest discover -s tests/host -v
```

Covered: legacy/explicit configuration precedence, default synchronous behavior,
thread-affine RPC creation/use/close, pending versus miss, ordered cancellation,
admitted lease ownership, manager/factory delegation, snapshot privacy/budget,
shared Attention/state pending counts, both-rank ACK timing, fence-before-source
release, and fail-closed quarantine/shutdown. The ordinary integration test that
previously expected multimodal rejection is updated but was not run without its
full dependency environment.

## Remaining exact-commit acceptance checks

Keep Draft until the repository's real LMCache/vLLM bootstrap is confirmed and
a focused NPU smoke runs on the exact updated commit. Record loaded source and
package versions, prove one CPU restore with zero local hits and both-rank
completion, check cold versus restored output with readable deterministic input,
then cover a mixed image/text/restore step and cancellation/shutdown boundaries.
Full queue pressure, real DMA/driver errors and graph/MTP combinations still need
hardware integration coverage. No need to discard or blindly repeat the entire
existing performance matrix to establish this source-to-runtime link.
