# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
from typing import Any, Iterable, Optional
import contextlib
import csv
import gc
import json
import os
import sys
import time

# Third Party
# TokensPrompt is unused in the final implementation but retained from the
# original benchmark imports.
from absl import logging
from hole_probe_utils import (
    DEFAULT_LAYER_TIMER_LAYERS,
    aggregate_layer_timer_file,
    build_empty_layer_timer_metrics,
)
from hydra.core.hydra_config import HydraConfig
from lmcache.logging import init_logger
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
import hydra
import numpy as np
import pandas as pd
import torch
import tqdm

os.environ.setdefault("PYTHONHASHSEED", "2026")

logger = init_logger(__name__)
try:
    # Third Party
    from lmcache.v1.trace_utils import trace_flow, trace_request_selected
except Exception:  # pragma: no cover - tracing is optional

    def trace_flow(*args, **kwargs):
        return None

    def trace_request_selected(*args, **kwargs) -> bool:
        return False


# get rid of [absl][INFO] - Using default tokenizer.
logging.set_verbosity(logging.WARNING)
# --- Utility Context Managers and Classes (Kept as is) ---

_STREAM_TEE_FILE = None
_STREAM_TEE_PATH = None
_STREAM_TEE_INSTALLED = False


class _StreamTee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


def _enable_benchmark_log_tee() -> None:
    """Mirror stdout/stderr into Hydra run-dir benchmark-hole.log."""
    global _STREAM_TEE_FILE, _STREAM_TEE_INSTALLED, _STREAM_TEE_PATH
    if _STREAM_TEE_INSTALLED:
        return

    run_dir = HydraConfig.get().run.dir
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "benchmark-hole.log")
    _STREAM_TEE_FILE = open(log_path, "a", buffering=1, encoding="utf-8")
    _STREAM_TEE_PATH = log_path
    sys.stdout = _StreamTee(sys.stdout, _STREAM_TEE_FILE)
    sys.stderr = _StreamTee(sys.stderr, _STREAM_TEE_FILE)
    _STREAM_TEE_INSTALLED = True
    print(f"[benchmark-hole] stream tee enabled: {log_path}", flush=True)


def get_lmcache_connector_spec(enable_holes: bool) -> tuple[str, str]:
    if enable_holes:
        return (
            "LMCacheAscendHoleConnectorV1Dynamic",
            "lmcache_ascend.integration.vllm.lmcache_ascend_hole_connector_v1",
        )
    return (
        "LMCacheAscendConnectorV1Dynamic",
        "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
    )


@contextlib.contextmanager
def build_llm_with_lmcache(
    model: str,
    tp_size: int,
    max_model_len: int = 32000,
    bs: int = 1,
    enable_holes: bool = False,
):
    connector_name, connector_module_path = get_lmcache_connector_spec(enable_holes)
    ktc = KVTransferConfig(
        kv_connector=connector_name,
        kv_role="kv_both",
        kv_connector_module_path=connector_module_path,
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        max_num_seqs=bs,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        _shutdown_llm_instance(llm)


@contextlib.contextmanager
def build_llm_with_vllm(
    model: str,
    tp_size: int,
    max_model_len: int = 32000,
    bs: int = 1,
    enable_holes: bool = False,
):
    # Vanilla vLLM does not require KVTransferConfig

    llm_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        max_num_seqs=bs,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        _shutdown_llm_instance(llm)


def _shutdown_llm_instance(llm: Optional[LLM]) -> None:
    if llm is None:
        return

    # vLLM's offline LLM API does not expose a public close() method, but the
    # engine core client does have an explicit shutdown path and upstream tests
    # call it directly. Relying on GC is too weak for repeated benchmark runs.
    try:
        llm_engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()
    except Exception:
        logger.exception("Failed to shut down vLLM engine core cleanly.")

    llm = None
    gc.collect()

    if hasattr(torch, "npu"):
        with contextlib.suppress(Exception):
            torch.npu.synchronize()
        with contextlib.suppress(Exception):
            torch.npu.empty_cache()

    shutdown_wait_s = float(os.environ.get("LMCACHE_LLM_SHUTDOWN_WAIT_S", "0.5"))
    if shutdown_wait_s > 0:
        time.sleep(shutdown_wait_s)


def loadmqa(data_path: str):
    data_path_l = data_path.lower()

    if data_path_l.endswith(".parquet"):
        # code not ready
        df = pd.read_parquet(data_path)
        ds = []
        for item in df.to_dict("records"):
            ds.append(
                {
                    "ctxs": list(item["ctxs"]),
                    "question": item["question"],
                    "answers": list(item["answers"]),
                }
            )
        return ds
    if data_path_l.endswith(".jsonl"):
        ds = []
        with open(data_path, encoding="utf-8") as fs:
            for line_no, line in enumerate(fs, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ds.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {data_path}:{line_no}: {exc}"
                    ) from exc
        return ds

    with open(data_path, encoding="utf-8") as fs:
        try:
            return json.load(fs)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse dataset {data_path} as JSON. "
                "If this is JSONL, use a .jsonl filename/extension."
            ) from exc


def dump_res(ds):
    output_dir = HydraConfig.get().run.dir
    with open(output_dir + "/res.json", "w") as f:
        json.dump(ds, f, indent=2)


def dump_answer_quality(ds):
    output_dir = HydraConfig.get().run.dir
    out_path = output_dir + "/answer_quality.tsv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["idx", "predicted", "ground_truth", "f1", "question"])
        for row in ds:
            if not isinstance(row, dict):
                continue
            if "generated" not in row:
                continue
            writer.writerow(
                [
                    row.get("idx", ""),
                    row.get("generated", ""),
                    json.dumps(row.get("ground truth", ""), ensure_ascii=False),
                    row.get("f1", ""),
                    row.get("question", ""),
                ]
            )


def plot_query_latency_hit_rate(phase_results: list[tuple[str, list]]) -> str | None:
    output_dir = HydraConfig.get().run.dir
    out_path = os.path.join(output_dir, "phase2_query_latency_hit_rate.png")
    try:
        # Third Party
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping plot generation (matplotlib unavailable): %s", exc)
        return None

    if not phase_results:
        logger.warning("No phase results available for plotting.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plotted_any = False

    for phase_name, phase_output in phase_results:
        query_rows = [
            row
            for row in phase_output
            if isinstance(row, dict) and "latency_e2e_s" in row
        ]
        if not query_rows:
            continue
        plotted_any = True
        x = list(range(len(query_rows)))
        latency = [
            (
                float(row["latency_e2e_s"])
                if row.get("latency_e2e_s", None) is not None
                else np.nan
            )
            for row in query_rows
        ]
        hit_rate = [
            (
                float(row["cache_hit_rate"])
                if row.get("cache_hit_rate", None) is not None
                else np.nan
            )
            for row in query_rows
        ]

        axes[0].plot(x, latency, marker="o", linewidth=1.5, label=phase_name)
        axes[1].plot(x, hit_rate, marker="o", linewidth=1.5, label=phase_name)

    if not plotted_any:
        plt.close(fig)
        logger.warning("No per-query rows found for plotting.")
        return None

    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Per-query E2E latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Query index")
    axes[1].set_ylabel("Hit rate")
    axes[1].set_title("Per-query cache hit rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(out_path)


def get_all_str(d):
    if isinstance(d, list):
        res = []
        for s in d:
            res += get_all_str(s)
        return res
    else:
        assert isinstance(d, str)
        res = [d]
        return res


def _extract_metric_number(container, names: list[str]) -> float | None:
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    candidates = [container]
    for attr in (
        "metrics",
        "extra",
        "extra_metrics",
        "custom_metrics",
        "connector_metrics",
        "kv_transfer_params",
    ):
        nested = getattr(container, attr, None)
        if nested is not None:
            candidates.append(nested)

    for item in candidates:
        if isinstance(item, dict):
            for name in names:
                if name in item:
                    converted = _to_float(item.get(name))
                    if converted is not None:
                        return converted
        for name in names:
            converted = _to_float(getattr(item, name, None))
            if converted is not None:
                return converted
    return None


def extract_cached_tokens(output) -> int | None:
    value = _extract_metric_number(
        output,
        ["num_cached_tokens", "cached_tokens", "lmcache_cached_tokens"],
    )
    if value is None:
        return None
    return int(value)


def extract_request_hit_tokens(output) -> int | None:
    value = _extract_metric_number(
        output,
        ["lmcache_hit_tokens", "hit_tokens"],
    )
    if value is None:
        return None
    return int(value)


def extract_request_prompt_tokens(output) -> int | None:
    value = _extract_metric_number(
        output,
        ["lmcache_prompt_tokens", "prompt_tokens"],
    )
    if value is None:
        return None
    return int(value)


def extract_request_hit_rate(output) -> float | None:
    return _extract_metric_number(
        output,
        [
            "req_hit_rate",
            "lmcache_req_hit_rate",
            "lmcache.req_hit_rate",
            "lmcache_hole_req_hit_rate",
        ],
    )


def extract_request_e2e_s(output) -> float | None:
    value = _extract_metric_number(
        output,
        [
            "time_to_completion",
            "e2e_latency",
            "end_to_end_latency",
            "latency",
            "request_latency",
        ],
    )
    if value is None or value < 0.0:
        return None
    return float(value)


def extract_request_ttft_s(output) -> float | None:
    value = _extract_metric_number(
        output,
        [
            "time_to_first_token",
            "ttft",
            "time_to_first_token_seconds",
        ],
    )
    if value is None or value < 0.0:
        return None
    return float(value)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=float)))


def _safe_percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), pct))


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.duration = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return False


def _resolve_submit_mode(raw_value: object, key_name: str) -> str:
    mode = str(raw_value).strip().lower().replace("-", "_")
    if mode in ("all_at_once", "allatonce", "all"):
        return "all_at_once"
    if mode in ("one_by_one", "onebyone", "one"):
        return "one_by_one"
    raise ValueError(
        f"Unsupported {key_name}={raw_value}. Expected one of: all_at_once, one_by_one."
    )


def _env_enabled(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default)).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _run_generate_requests(
    llm: LLM,
    request_inputs: list[dict[str, list[int]]],
    sampling_params: SamplingParams,
    submit_mode: str,
    phase_label: str,
):
    if submit_mode == "all_at_once":
        return llm.generate(
            request_inputs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

    outputs = []
    for request in tqdm.tqdm(
        request_inputs,
        desc=f"{phase_label} one_by_one",
    ):
        req_outputs = llm.generate(
            [request],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        outputs.extend(req_outputs)
    return outputs


def _iter_metric_containers(output: Any) -> list[Any]:
    containers: list[Any] = [output]

    metrics = getattr(output, "metrics", None)
    if metrics is not None:
        containers.append(metrics)
        for attr in ("extra", "extra_metrics", "custom_metrics", "connector_metrics"):
            nested = getattr(metrics, attr, None)
            if nested is not None:
                containers.append(nested)

    kv_params = getattr(output, "kv_transfer_params", None)
    if kv_params is not None:
        containers.append(kv_params)

    return containers


def extract_text_metric(output: Any, names: Iterable[str]) -> Optional[str]:
    for container in _iter_metric_containers(output):
        if isinstance(container, dict):
            for name in names:
                value = container.get(name)
                if value is not None:
                    return str(value)

        for name in names:
            value = getattr(container, name, None)
            if value is not None:
                return str(value)

    return None


def extract_request_mode(output: Any) -> Optional[str]:
    value = extract_text_metric(output, ("lmcache_mode", "mode"))
    if value is None:
        return None
    return value.strip()


def ensure_hole_probe_mode(output: Any) -> str:
    mode = extract_request_mode(output)
    if mode != "hole":
        raise RuntimeError(
            f"Hole-mode preflight probe expected lmcache_mode='hole', got {mode!r}."
        )
    return mode


def _load_request_layer_timer_metrics(
    request_ids: Iterable[Optional[str]],
) -> dict[str, dict[str, Any]]:
    if _STREAM_TEE_FILE is None or _STREAM_TEE_PATH is None:
        return {}

    filtered_req_ids = [
        str(request_id) for request_id in request_ids if request_id is not None
    ]
    if not filtered_req_ids:
        return {}

    _STREAM_TEE_FILE.flush()
    try:
        return aggregate_layer_timer_file(
            _STREAM_TEE_PATH,
            req_ids=filtered_req_ids,
            layers=DEFAULT_LAYER_TIMER_LAYERS,
        )
    except OSError as exc:
        logger.warning(
            "Failed to load layer timer metrics from %s: %s", _STREAM_TEE_PATH, exc
        )
        return {}


def _run_hole_mode_preflight(
    llm: LLM,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    preinst_ids: torch.Tensor,
    postinst_ids: torch.Tensor,
    prefix_prompt: str,
) -> None:
    logger.info("Running hole-mode preflight probe.")

    def tokenize1(prompt: str):
        return (
            tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            .to(torch.long)
            .reshape(1, -1)
        )

    def make_doc_tokens(doc_idx: int, min_tokens: int) -> torch.Tensor:
        repeats = max(min_tokens, 32)
        doc_text = " ".join([f"holeprobe{doc_idx}"] * repeats)
        doc_ids = tokenize1(doc_text).cpu()
        while doc_ids.numel() <= min_tokens:
            repeats *= 2
            doc_text = " ".join([f"holeprobe{doc_idx}"] * repeats)
            doc_ids = tokenize1(doc_text).cpu()
        return doc_ids

    sep = tokenize1(cfg.sep).cpu()
    s_start_full = torch.cat(
        (preinst_ids.cpu(), tokenize1(prefix_prompt).cpu()),
        dim=1,
    ).to(torch.long)
    min_segment_tokens = max(int(getattr(cfg, "chunk_size", 256)) + 32, 320)
    doc_ids = [make_doc_tokens(doc_idx, min_segment_tokens) for doc_idx in range(5)]
    q_ids = torch.cat(
        (
            tokenize1("Hole probe question?\nAnswer:").cpu(),
            postinst_ids.cpu(),
        ),
        dim=1,
    ).to(torch.long)

    prefill_requests = [
        {
            "prompt_token_ids": (
                s_start_full.reshape(-1).tolist() + sep.reshape(-1).tolist()
            )
        }
    ]
    for doc_idx in (0, 2, 4):
        prefill_requests.append(
            {
                "prompt_token_ids": (
                    doc_ids[doc_idx].reshape(-1).tolist() + sep.reshape(-1).tolist()
                )
            }
        )

    prefill_sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        extra_args={"temperature": 1.0, "prefix": 0},
    )
    _run_generate_requests(
        llm=llm,
        request_inputs=prefill_requests,
        sampling_params=prefill_sampling_params,
        submit_mode="one_by_one",
        phase_label="hole_probe_prefill",
    )

    llm.reset_prefix_cache(reset_running_requests=True, reset_connector=False)

    probe_tokens = [s_start_full, sep]
    for doc_id in doc_ids:
        probe_tokens.append(doc_id)
        probe_tokens.append(sep)
    probe_tokens.append(q_ids)

    probe_request = {
        "prompt_token_ids": torch.cat(probe_tokens, dim=1).reshape(-1).tolist()
    }
    probe_sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        extra_args={"temperature": 1.0, "prefix": s_start_full.numel()},
    )
    probe_outputs = _run_generate_requests(
        llm=llm,
        request_inputs=[probe_request],
        sampling_params=probe_sampling_params,
        submit_mode="one_by_one",
        phase_label="hole_probe_query",
    )
    probe_output = probe_outputs[0]
    probe_mode = extract_request_mode(probe_output)
    logger.warning(
        "[LMC-DEBUG][hole-probe] mode=%s hit_tokens=%s prompt_tokens=%s",
        probe_mode,
        extract_request_hit_tokens(probe_output),
        extract_request_prompt_tokens(probe_output),
    )
    ensure_hole_probe_mode(probe_output)
    llm.reset_prefix_cache(reset_running_requests=True, reset_connector=False)


def _prepare_query_docs(
    ex: dict,
    cfg: DictConfig,
    build_qa_prompt_fn,
    query_prompt: str,
) -> tuple[list[str], str]:
    doc_prompts, suffix_prompt = build_qa_prompt_fn(ex, query_prompt)
    if cfg.cap_ndocs >= 0:
        doc_prompts = doc_prompts[: cfg.cap_ndocs]
    cap_doc_chars = int(getattr(cfg, "cap_doc_chars", -1))
    if cap_doc_chars >= 0:
        doc_prompts = [str(doc)[:cap_doc_chars] for doc in doc_prompts]
    return doc_prompts, suffix_prompt


def _resolve_hole_pos(hole_pos: int, total_docs: int) -> int:
    if total_docs <= 0:
        return -1

    resolved_hole_pos = hole_pos
    if resolved_hole_pos < 0:
        resolved_hole_pos = total_docs // 2
    if resolved_hole_pos >= total_docs:
        raise ValueError(
            f"hole_pos={hole_pos} out of range for inflated query docs len={total_docs}"
        )
    return resolved_hole_pos


def _resolve_hole_id(hole_id: int, native_doc_count: int) -> int:
    if native_doc_count <= 0:
        return -1

    resolved_hole_id = hole_id
    if resolved_hole_id < 0:
        resolved_hole_id = native_doc_count // 2
    if resolved_hole_id >= native_doc_count:
        raise ValueError(
            f"hole_id={hole_id} out of range for native query docs "
            f"len={native_doc_count}"
        )
    return resolved_hole_id


def _coerce_int_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            parsed = json.loads(text)
            return [int(item) for item in parsed]
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
        return [int(item) for item in raw]
    return [int(raw)]


def _resolve_hole_lists(cfg: DictConfig) -> tuple[list[int], list[int]]:
    hole_ids = _coerce_int_list(getattr(cfg, "hole_ids", []))
    hole_positions = _coerce_int_list(getattr(cfg, "hole_positions", []))
    if bool(hole_ids) != bool(hole_positions):
        raise ValueError("hole_ids and hole_positions must be set together")
    if hole_ids:
        if len(hole_ids) != len(hole_positions):
            raise ValueError("hole_ids and hole_positions must have the same length")
        return hole_ids, hole_positions
    return [int(getattr(cfg, "hole_id", -1))], [int(getattr(cfg, "hole_pos", -1))]


def _resolve_hole_specs(
    *,
    hole_ids: list[int],
    hole_positions: list[int],
    native_doc_count: int,
    total_doc_count: int,
) -> tuple[list[int], list[int]]:
    resolved_hole_ids = [
        _resolve_hole_id(hole_id, native_doc_count) for hole_id in hole_ids
    ]
    resolved_hole_positions = [
        _resolve_hole_pos(hole_pos, total_doc_count) for hole_pos in hole_positions
    ]
    if len(set(resolved_hole_ids)) != len(resolved_hole_ids):
        raise ValueError(f"Resolved hole_ids contain duplicates: {resolved_hole_ids}")
    if len(set(resolved_hole_positions)) != len(resolved_hole_positions):
        raise ValueError(
            f"Resolved hole_positions contain duplicates: {resolved_hole_positions}"
        )
    return resolved_hole_ids, resolved_hole_positions


def _build_selected_entries(
    eval_dataset: list,
    cfg: DictConfig,
    build_qa_prompt_fn,
    query_prompt: str,
    hole_ids: list[int],
    hole_positions: list[int],
    inflate: int,
) -> list[dict[str, Any]]:
    selected_entries: list[dict[str, Any]] = []
    select_cnt = 0

    for idx, ex in enumerate(eval_dataset[: cfg.until]):
        if not (
            len(cfg.only_s) == 0
            or any(s in cfg.only_s for s in get_all_str(ex["answers"]))
        ):
            continue
        if len(cfg.only_s) > 0:
            if select_cnt == 1:
                break
            select_cnt += 1

        doc_prompts, suffix_prompt = _prepare_query_docs(
            ex=ex,
            cfg=cfg,
            build_qa_prompt_fn=build_qa_prompt_fn,
            query_prompt=query_prompt,
        )
        native_doc_refs = [
            (int(idx), int(doc_i), str(doc_prompt))
            for doc_i, doc_prompt in enumerate(doc_prompts)
        ]
        selected_entries.append(
            {
                "idx": int(idx),
                "answers": ex["answers"],
                "suffix_prompt": suffix_prompt,
                "native_doc_refs": native_doc_refs,
            }
        )

    all_doc_refs = [
        doc_ref for row in selected_entries for doc_ref in row["native_doc_refs"]
    ]
    reserved_hole_refs: set[tuple[int, int, str]] = set()

    for row in selected_entries:
        native_doc_refs = list(row["native_doc_refs"])
        total_doc_count = len(native_doc_refs) + inflate
        resolved_hole_ids, resolved_hole_positions = _resolve_hole_specs(
            hole_ids=hole_ids,
            hole_positions=hole_positions,
            native_doc_count=len(native_doc_refs),
            total_doc_count=total_doc_count,
        )
        hole_doc_refs = [
            native_doc_refs[resolved_hole_id] for resolved_hole_id in resolved_hole_ids
        ]
        row["resolved_hole_ids"] = resolved_hole_ids
        row["resolved_hole_positions"] = resolved_hole_positions
        row["hole_doc_refs"] = hole_doc_refs
        row["resolved_hole_id"] = (
            resolved_hole_ids[0] if len(resolved_hole_ids) == 1 else None
        )
        row["resolved_hole_pos"] = (
            resolved_hole_positions[0] if len(resolved_hole_positions) == 1 else None
        )
        row["hole_doc_ref"] = hole_doc_refs[0] if len(hole_doc_refs) == 1 else None
        reserved_hole_refs.update(hole_doc_refs)

    for row in selected_entries:
        row_idx = int(row["idx"])
        native_doc_refs = list(row["native_doc_refs"])
        resolved_hole_ids = [int(item) for item in row["resolved_hole_ids"]]
        resolved_hole_positions = [int(item) for item in row["resolved_hole_positions"]]
        hole_doc_refs = list(row["hole_doc_refs"])
        candidate_extra_refs = [
            doc_ref
            for doc_ref in all_doc_refs
            if doc_ref[0] != row_idx and doc_ref not in reserved_hole_refs
        ]
        if len(candidate_extra_refs) < inflate:
            raise ValueError(
                f"inflate={inflate} requires {inflate} non-hole extra docs "
                f"for query idx={row_idx}, but only "
                f"{len(candidate_extra_refs)} are available"
            )

        extra_doc_refs = list(candidate_extra_refs[:inflate])
        base_final_doc_refs = native_doc_refs + extra_doc_refs
        hole_doc_ref_set = set(hole_doc_refs)
        remaining_doc_refs = [
            doc_ref
            for doc_ref in base_final_doc_refs
            if doc_ref not in hole_doc_ref_set
        ]
        final_doc_refs: list[tuple[int, int, str] | None] = [None] * len(
            base_final_doc_refs
        )
        for resolved_hole_pos, hole_doc_ref in zip(
            resolved_hole_positions,
            hole_doc_refs,
            strict=False,
        ):
            final_doc_refs[resolved_hole_pos] = hole_doc_ref
        remaining_iter = iter(remaining_doc_refs)
        for i, doc_ref in enumerate(final_doc_refs):
            if doc_ref is None:
                final_doc_refs[i] = next(remaining_iter)
        if any(doc_ref is None for doc_ref in final_doc_refs):
            raise AssertionError("final_doc_refs contains unfilled entries")
        resolved_final_doc_refs = [
            doc_ref for doc_ref in final_doc_refs if doc_ref is not None
        ]

        row["doc_prompts"] = [doc_ref[2] for doc_ref in native_doc_refs]
        row["extra_doc_refs"] = list(resolved_final_doc_refs[len(native_doc_refs) :])
        row["extra_doc_prompts"] = [doc_ref[2] for doc_ref in row["extra_doc_refs"]]
        row["final_doc_refs"] = resolved_final_doc_refs
        row["final_doc_prompts"] = [doc_ref[2] for doc_ref in resolved_final_doc_refs]

    return selected_entries


def _format_hole_summary(selected_entries: list[dict[str, Any]]) -> str:
    parts = [
        (
            f"{int(row['idx'])}:"
            + "|".join(
                f"{int(hole_id)}@{int(hole_pos)}"
                for hole_id, hole_pos in zip(
                    row["resolved_hole_ids"],
                    row["resolved_hole_positions"],
                    strict=False,
                )
            )
        )
        for row in selected_entries
    ]
    return f"holes:[{','.join(parts)}]"


# --- Prefill Function (Fixed input format for batched generation) ---


def prefill_chunks(
    llm: LLM,
    eval_dataset: list,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    preinst_ids: torch.Tensor,
    prefix_prompt: str,
    hole_mode: str,
    hole_ids: list[int],
    hole_positions: list[int],
    phase1_submit_mode: str,
    build_qa_prompt_fn,
):
    """
    Step 1: Pre-populate KV caches for all document chunks in one batch.
    """
    logger.info("Starting batched document prefilling...")

    # Define tokenization helper (using CPU tensors for batch preparation)
    def tokenize1(prompt: str):
        return (
            tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            .to(torch.long)
            .reshape(1, -1)
        )

    sep = tokenize1(cfg.sep).cpu()
    s_start_full = torch.cat(
        (preinst_ids.cpu(), tokenize1(prefix_prompt).cpu()), dim=1
    ).to(torch.long)
    nosink = bool(getattr(cfg, "nosink", False))

    # Shared prefix for Golem/CacheBlend (if configured)
    def build_shared_prefix(b):
        if b == 0:
            return None
        shared_prompt_s = "You are a Q&A assistant. "
        if b == -1:
            return tokenize1(shared_prompt_s)
        return tokenize1(" " * b)

    shared_prefix_t = (
        build_shared_prefix(int(cfg.golem_prefix)).cpu()
        if int(cfg.golem_prefix) != 0
        else None
    )

    all_prefill_prompts = []
    inflate = int(getattr(cfg, "inflate", 1))
    selected_entries = _build_selected_entries(
        eval_dataset=eval_dataset,
        cfg=cfg,
        build_qa_prompt_fn=build_qa_prompt_fn,
        query_prompt="",
        hole_ids=hole_ids,
        hole_positions=hole_positions,
        inflate=inflate,
    )
    reserved_hole_refs = {
        hole_doc_ref
        for row in selected_entries
        for hole_doc_ref in row.get("hole_doc_refs", [])
    }
    # Build batched prefill tasks from every query and document.
    for row in tqdm.tqdm(selected_entries, desc="Preparing prefill prompts"):
        idx = int(row["idx"])
        doc_prompts = list(row["doc_prompts"])
        docs_only_ids = [tokenize1(doc).cpu() for doc in doc_prompts]

        # No multi-doc super-chunk semantics: each unit is prefetched independently.
        if nosink:
            # Sink-removal prefill shape:
            # sysprompt + sep + doc + sep (for each prefilled doc).
            for doc_i, doc_ids in enumerate(docs_only_ids):
                doc_ref = (idx, doc_i, str(doc_prompts[doc_i]))
                if doc_ref in reserved_hole_refs:
                    continue
                prefill_tensor = torch.cat((s_start_full, sep, doc_ids, sep), dim=1)
                all_prefill_prompts.append(prefill_tensor.reshape(-1).tolist())
        else:
            docs_ids = [s_start_full] + docs_only_ids
            for i, onedoc_ids in enumerate(docs_ids):
                if i > 0:
                    doc_ref = (idx, i - 1, str(doc_prompts[i - 1]))
                else:
                    doc_ref = None
                if doc_ref is not None and doc_ref in reserved_hole_refs:
                    continue

                prompt_token_ids = onedoc_ids.reshape(-1).tolist()

                # Apply prepend logic from original code: shared_prefix + sep (if i > 0)
                if shared_prefix_t is not None and i > 0:
                    prompt_token_ids = (
                        shared_prefix_t.reshape(-1).tolist()
                        + sep.reshape(-1).tolist()
                        + prompt_token_ids
                    )

                # Apply postfix logic from original code: sep
                prompt_token_ids = prompt_token_ids + sep.reshape(-1).tolist()
                all_prefill_prompts.append(prompt_token_ids)

    # 2. Run batched prefilling
    if not all_prefill_prompts:
        logger.warning("No prefill chunks generated. Skipping prefilling.")
        return {
            "phase_label": "phase1_prefill",
            "total_prefill_time_s": 0.0,
            "num_prefill_kv_caches": 0,
            "avg_prefill_kv_cache_s": None,
        }

    logger.info(
        "Total %d document chunks to prefill (submit_mode=%s, nosink=%s).",
        len(all_prefill_prompts),
        phase1_submit_mode,
        nosink,
    )

    prefill_sampling_params = SamplingParams(
        max_tokens=1, temperature=0.0, extra_args={"temperature": 1.0, "prefix": 0}
    )

    # FIX: Use list of request dictionaries for vLLM batch input
    request_inputs = [{"prompt_token_ids": tokens} for tokens in all_prefill_prompts]

    with Timer() as prefill_t:
        _run_generate_requests(
            llm=llm,
            request_inputs=request_inputs,
            sampling_params=prefill_sampling_params,
            submit_mode=phase1_submit_mode,
            phase_label="phase1_prefill",
        )

    avg_prefill_kv_cache_s = prefill_t.duration / len(all_prefill_prompts)
    logger.info(
        "Document prefilling completed in %.2f seconds "
        "(avg %.4f s per KV cache over %d prefills).",
        prefill_t.duration,
        avg_prefill_kv_cache_s,
        len(all_prefill_prompts),
    )
    return {
        "phase_label": "phase1_prefill",
        "total_prefill_time_s": float(prefill_t.duration),
        "num_prefill_kv_caches": int(len(all_prefill_prompts)),
        "avg_prefill_kv_cache_s": float(avg_prefill_kv_cache_s),
    }


# --- Updated run_batched_queries function with Latency Metrics ---


def run_batched_queries(
    llm: LLM,
    eval_dataset: list,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    preinst_ids: torch.Tensor,
    postinst_ids: torch.Tensor,
    prefix_prompt: str,
    query_prompt: str,
    build_qa_prompt_fn,
    compute_f1_fn,
    compute_rl_fn,
    phase2_submit_mode: str,
    phase_label: str = "phase2",
) -> list:
    """
    Step 2: Run final query generation and collect latency metrics.
    """
    logger.info("Starting query generation for %s...", phase_label)
    latency_warn = False
    inflate = int(getattr(cfg, "inflate", 1))
    if inflate < 0:
        raise ValueError("inflate must be >= 0")
    hole_ids, hole_positions = _resolve_hole_lists(cfg)

    # Define tokenization helper (using cpu tensors for batch preparation)
    def tokenize1(prompt: str):
        return (
            tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            .to(torch.long)
            .reshape(1, -1)
        )

    all_query_prompts = []
    metadata_list = []

    s_start_full = torch.cat(
        (preinst_ids.cpu(), tokenize1(prefix_prompt).cpu()), dim=1
    ).to(torch.long)
    sep = tokenize1(cfg.sep).cpu()

    selected_entries = _build_selected_entries(
        eval_dataset=eval_dataset,
        cfg=cfg,
        build_qa_prompt_fn=build_qa_prompt_fn,
        query_prompt=query_prompt,
        hole_ids=hole_ids,
        hole_positions=hole_positions,
        inflate=inflate,
    )
    logger.info(_format_hole_summary(selected_entries))

    logger.info(
        "Using inflate=%d: hole_ids select native docs to omit from prefill, "
        "and hole_positions place those docs into the inflated prompt.",
        inflate,
    )

    # 2. Build final query prompts.
    for row in tqdm.tqdm(selected_entries, desc="Preparing final query prompts"):
        idx = int(row["idx"])
        answers = row["answers"]
        suffix_prompt = row["suffix_prompt"]
        final_doc_prompts = list(row["final_doc_prompts"])

        try:
            docs_ids = [tokenize1(doc).cpu() for doc in final_doc_prompts]
            q_ids = torch.cat(
                (tokenize1(suffix_prompt).cpu(), postinst_ids.cpu()), dim=1
            ).to(torch.long)

            # Build the sequence:
            # S_START_FULL + SEP + final inflated docs + SEP + Q_IDS
            all_tokens = [s_start_full]
            all_tokens.append(sep)
            for doc_id in docs_ids:
                all_tokens.append(doc_id)
                all_tokens.append(sep)

            # Append the final question part
            all_tokens.append(q_ids)

            all_ids = torch.cat(all_tokens, dim=1).reshape(-1).tolist()
            all_query_prompts.append(all_ids)

            # Store metadata for post-processing
            metadata_list.append(
                {
                    "idx": idx,
                    "answers": answers,
                    "question": suffix_prompt,
                    "s_start_full_len": s_start_full.numel(),
                    "resolved_hole_ids": list(row["resolved_hole_ids"]),
                    "resolved_hole_positions": list(row["resolved_hole_positions"]),
                    "final_doc_count": len(final_doc_prompts),
                }
            )

        except Exception as e:
            logger.error(f"Error preparing query {idx}: {e}")

    if not all_query_prompts:
        logger.warning("No final query prompts generated. Skipping generation.")
        return []

    logger.info(
        "Total %d queries to generate (submit_mode=%s).",
        len(all_query_prompts),
        phase2_submit_mode,
    )

    # 2. Run batched generation
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        extra_args={
            "temperature": cfg.golem_temperature,
            "prefix": s_start_full.numel(),
        },
    )

    query_inputs = [{"prompt_token_ids": tokens} for tokens in all_query_prompts]

    # We use a single Timer for the entire execution time.
    with Timer() as gen_t:
        outputs = _run_generate_requests(
            llm=llm,
            request_inputs=query_inputs,
            sampling_params=sampling_params,
            submit_mode=phase2_submit_mode,
            phase_label=phase_label,
        )

    # 3. Process results and metrics
    output_res = []
    f1_list = []
    rl_list = []

    # Metric lists
    e2e_times = []
    ttft_times = []
    tbt_times = []
    cached_tokens_total = 0
    cached_prompt_tokens_total = 0
    cached_metrics_samples = 0
    exported_hit_tokens_total = 0
    exported_prompt_tokens_total = 0
    exported_metrics_samples = 0

    if len(outputs) != len(metadata_list):
        logger.error(
            f"Mismatch between outputs ({len(outputs)}) and metadata "
            f"({len(metadata_list)})."
        )
        return []

    layer_timer_metrics_by_req = _load_request_layer_timer_metrics(
        getattr(output, "request_id", None) for output in outputs
    )
    timer_rows: list[dict[str, Any]] = []

    for output_idx, (output, meta) in enumerate(
        zip(outputs, metadata_list, strict=False)
    ):
        req_id = getattr(output, "request_id", None)
        timer_metrics = (
            layer_timer_metrics_by_req.get(
                str(req_id),
                build_empty_layer_timer_metrics(DEFAULT_LAYER_TIMER_LAYERS),
            )
            if req_id is not None
            else build_empty_layer_timer_metrics(DEFAULT_LAYER_TIMER_LAYERS)
        )
        timer_rows.append(timer_metrics)
        if not output.outputs:
            logger.warning(
                "Skipping request %s because vLLM returned no outputs.",
                req_id,
            )
            continue
        output_str = output.outputs[0].text
        answers = meta["answers"]

        e2e_time = extract_request_e2e_s(output)
        ttft = extract_request_ttft_s(output)
        tbt = None
        if e2e_time is not None and ttft is not None:
            generated_tokens = len(output.outputs[0].token_ids)
            tbt = (e2e_time - ttft) / max(1, generated_tokens - 1)

        if e2e_time is None and not latency_warn:
            logger.warning(
                "Per-request timing metrics are unavailable from vLLM output. "
                "TTFT/TBT will be left missing instead of approximated."
            )
            latency_warn = True

        if e2e_time is not None:
            e2e_times.append(e2e_time)
        if ttft is not None:
            ttft_times.append(ttft)
        if tbt is not None:
            tbt_times.append(tbt)

        cached_tokens = extract_cached_tokens(output)
        exported_hit_tokens = extract_request_hit_tokens(output)
        exported_prompt_tokens = extract_request_prompt_tokens(output)
        req_hit_rate = extract_request_hit_rate(output)
        prompt_tokens = (
            len(all_query_prompts[output_idx])
            if output_idx < len(all_query_prompts)
            else 0
        )
        per_query_hit_rate = None
        if exported_hit_tokens is not None and exported_prompt_tokens is not None:
            exported_hit_tokens_total += max(0, int(exported_hit_tokens))
            exported_prompt_tokens_total += max(0, int(exported_prompt_tokens))
            exported_metrics_samples += 1
            if exported_prompt_tokens > 0:
                per_query_hit_rate = max(
                    0.0,
                    min(
                        float(exported_hit_tokens) / float(exported_prompt_tokens),
                        1.0,
                    ),
                )
        elif cached_tokens is not None and output_idx < len(all_query_prompts):
            cached_tokens_total += max(0, int(cached_tokens))
            cached_prompt_tokens_total += prompt_tokens
            cached_metrics_samples += 1
            if prompt_tokens > 0:
                per_query_hit_rate = max(
                    0.0, min(float(cached_tokens) / float(prompt_tokens), 1.0)
                )
        if req_hit_rate is None:
            req_hit_rate = per_query_hit_rate

        # --- Accuracy Metrics ---
        f1 = compute_f1_fn(output_str, answers, tokenizer)
        f1_list.append(f1)
        rl = compute_rl_fn(output_str, answers)
        rl_list.append(rl)
        print(f"{output_str} vs {answers} ==> {f1}")

        # --- Output Dictionary ---
        d = {
            "idx": meta["idx"],
            "request_id": req_id,
            "f1": f1,
            "rl": rl,
            "ground truth": answers,
            "generated": output_str,
            "question": meta["question"],
            "latency_e2e_s": e2e_time,
            "latency_ttft_s": ttft,
            "latency_tbt_s": tbt,
            "prompt_tokens": int(prompt_tokens),
            "cached_tokens": int(cached_tokens) if cached_tokens is not None else None,
            "hit_tokens": int(exported_hit_tokens)
            if exported_hit_tokens is not None
            else None,
            "hit_prompt_tokens": int(exported_prompt_tokens)
            if exported_prompt_tokens is not None
            else None,
            "req_hit_rate": req_hit_rate,
            "cache_hit_rate": per_query_hit_rate,
        }
        d.update(timer_metrics)
        output_res.append(d)

        if trace_request_selected(req_id):
            trace_flow(
                "benchmark.output",
                "request_output",
                phase=phase_label,
                output_idx=output_idx,
                req_id=req_id,
                first_generated_token_id=(
                    int(output.outputs[0].token_ids[0])
                    if output.outputs and output.outputs[0].token_ids
                    else None
                ),
                generated_token_ids=[
                    int(token_id)
                    for token_id in getattr(output.outputs[0], "token_ids", [])
                ],
                generated_text=output_str,
                answers=answers,
                f1=float(f1),
            )

    # 4. Calculate Summary Metrics
    logger.info("%s generation completed in %.2f seconds.", phase_label, gen_t.duration)

    cache_hit_rate = None
    cache_hit_rate_source = "none"
    if exported_prompt_tokens_total > 0:
        cache_hit_rate = exported_hit_tokens_total / exported_prompt_tokens_total
        cache_hit_rate_source = "exported_hit_tokens"
    elif cached_prompt_tokens_total > 0:
        cache_hit_rate = cached_tokens_total / cached_prompt_tokens_total
        cache_hit_rate_source = "cached_tokens"
    elif cfg.method == "full":
        cache_hit_rate = 0.0
        cache_hit_rate_source = "baseline_zero"

    # Calculate Averages and Percentiles
    avg_e2e = _safe_mean(e2e_times)
    if avg_e2e is None and len(all_query_prompts):
        avg_e2e = gen_t.duration / len(all_query_prompts)

    summary_metrics = {
        "total_generation_time": gen_t.duration,
        "num_queries": len(all_query_prompts),
        "mean_f1": np.mean(f1_list) if f1_list else 0,
        "mean_rl": np.mean(rl_list) if rl_list else 0,
        "avg_e2e_s": avg_e2e,
        "p90_e2e_s": _safe_percentile(e2e_times, 90),
        "avg_ttft_s": _safe_mean(ttft_times),
        "p90_ttft_s": _safe_percentile(ttft_times, 90),
        "avg_tbt_s": _safe_mean(tbt_times),
        "p90_tbt_s": _safe_percentile(tbt_times, 90),
        "cache_hit_rate": cache_hit_rate,
        "cache_hit_rate_source": cache_hit_rate_source,
        "cached_tokens_total": cached_tokens_total,
        "cached_prompt_tokens_total": cached_prompt_tokens_total,
        "cached_metrics_samples": cached_metrics_samples,
        "exported_hit_tokens_total": exported_hit_tokens_total,
        "exported_prompt_tokens_total": exported_prompt_tokens_total,
        "exported_metrics_samples": exported_metrics_samples,
        "phase_label": phase_label,
        "timed_requests_e2e": len(e2e_times),
        "timed_requests_ttft": len(ttft_times),
        "timed_requests_tbt": len(tbt_times),
    }

    empty_timer_metrics = build_empty_layer_timer_metrics(DEFAULT_LAYER_TIMER_LAYERS)
    for bucket in ("wait_reuse", "topk_l1", "blend", "save"):
        field = f"lmcache_timer_{bucket}_ms"
        total_field = f"lmcache_timer_{bucket}_total_ms"
        if timer_rows:
            values = np.array([row[field] for row in timer_rows], dtype=float)
            totals = np.array([row[total_field] for row in timer_rows], dtype=float)
            summary_metrics[f"avg_{field}"] = values.mean(axis=0).tolist()
            summary_metrics[f"avg_{total_field}"] = float(totals.mean())
        else:
            summary_metrics[f"avg_{field}"] = empty_timer_metrics[field]
            summary_metrics[f"avg_{total_field}"] = 0.0

    # Add summary to results (always the last element)
    output_res.append(summary_metrics)

    return output_res


# --- Main Function (Kept as is) ---


@hydra.main(config_path="config", config_name="config-hole.yaml", version_base="1.3")
def main(cfg: DictConfig):
    hole_mode_raw = str(getattr(cfg, "hole_mode", "no-hole")).strip().lower()
    if hole_mode_raw in ("hole", "1", "true", "yes", "on"):
        hole_mode = "hole"
    elif hole_mode_raw in ("no-hole", "nohole", "0", "false", "no", "off"):
        hole_mode = "no-hole"
    else:
        raise ValueError(f"Unsupported hole_mode={hole_mode_raw}")
    hole_ids, hole_positions = _resolve_hole_lists(cfg)
    max_num_req_raw = getattr(cfg, "max_num_req", None)
    max_num_req = None
    if max_num_req_raw is not None:
        max_num_req = int(max_num_req_raw)
        if max_num_req <= 0:
            raise ValueError("max_num_req must be > 0 when set")
    max_num_reqs = max_num_req if max_num_req is not None else int(cfg.batch_size)
    if max_num_reqs <= 0:
        raise ValueError("max_num_reqs must be > 0")
    phase2_repeats = int(getattr(cfg, "phase2_repeats", 2))
    if phase2_repeats <= 0:
        raise ValueError("phase2_repeats must be > 0")
    phase1_submit_mode = _resolve_submit_mode(
        getattr(cfg, "phase1_submit_mode", "all_at_once"),
        "phase1_submit_mode",
    )
    phase2_submit_mode = _resolve_submit_mode(
        getattr(cfg, "phase2_submit_mode", "all_at_once"),
        "phase2_submit_mode",
    )
    cap_doc_chars = int(getattr(cfg, "cap_doc_chars", -1))
    if cap_doc_chars < -1:
        raise ValueError("cap_doc_chars must be >= -1")

    _enable_benchmark_log_tee()
    if cap_doc_chars >= 0:
        logger.info("Capping each document prompt to %d characters.", cap_doc_chars)

    # Set environment variables
    os.environ["LMCACHE_CHUNK_SIZE"] = str(int(getattr(cfg, "chunk_size", 256)))
    os.environ["VLLM_ENFORCE_EAGER"] = "True"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(cfg.recompute_ratio)
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = cfg.sep
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "False"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ.setdefault("LMCACHE_SAVE_UNFULL_CHUNK", "true")
    os.environ.setdefault("LMCACHE_TRACE_BLEND", "0")
    os.environ.setdefault("LMCACHE_TRACE_FLOW", "0")
    os.environ.setdefault("LMCACHE_TRACE_MAX_ITEMS", "256")
    os.environ.setdefault("LMCACHE_ENABLE_LAYER_TIMERS", "1")
    os.environ.setdefault("LMCACHE_LAYER_TIMER_LAYERS", "0-4")
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "200"
    hole_enabled = hole_mode == "hole"
    os.environ["LMCACHE_ENABLE_HOLE_BLENDING"] = "1" if hole_enabled else "0"

    # IMPORTANT: lmcache_ascend import must happen after hole-mode env setup,
    # otherwise hole monkey patches are decided with default env values.
    # First Party
    from lmcache_ascend.utils import build_qa_prompt, compute_f1, compute_rl

    method2builder = {
        "cacheblend": build_llm_with_lmcache,
        "reuse": build_llm_with_lmcache,
        "full": build_llm_with_vllm,
    }

    os.environ["LMCACHE_ENABLE_BLENDING"] = (
        "True" if cfg.method == "cacheblend" else "False"
    )
    logger.warning(
        "[LMC-DEBUG][bench][mode] cfg.method=%s hole_mode=%s hole_ids=%s "
        "hole_positions=%s -> LMCACHE_ENABLE_BLENDING=%s",
        cfg.method,
        hole_mode,
        hole_ids,
        hole_positions,
        os.environ["LMCACHE_ENABLE_BLENDING"],
    )
    builder = method2builder[cfg.method]

    # Prepare common components
    eval_dataset = loadmqa(cfg.data_path)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(cfg.model_path)

    # Define tokenization helper for local tensors
    def tokenize1(prompt: str):
        return (
            tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            .to(torch.long)
            .reshape(1, -1)
        )

    # --- Pre-calculate fixed prompt tokens ---
    def get_inst_prefix_suffix_tokens() -> tuple[torch.Tensor, torch.Tensor]:
        user_content = "998244353"
        messages = [{"role": "user", "content": user_content}]
        # Use CPU for preparation
        full_encoded = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_special_tokens=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            .to(torch.long)
            .reshape(1, -1)
        )
        instruction_tokens = tokenize1(user_content)

        for i in range(full_encoded.shape[1] - instruction_tokens.shape[1] + 1):
            if (
                full_encoded[:, i : i + instruction_tokens.shape[1]]
                == instruction_tokens
            ).all():
                prefix = full_encoded[:, :i]
                suffix = full_encoded[:, i + instruction_tokens.shape[1] :]
                return prefix, suffix
        raise ValueError("Instruction tokens not found in templated token sequence.")

    preinst_ids, postinst_ids = get_inst_prefix_suffix_tokens()

    prefix_prompt = (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n"
    )
    query_prompt = (
        "\n\nAnswer the question based on the given passages. Answer the question "
        "within 5 words. Do NOT repeat the question or output any other words. "
        "Question: "
    )
    phase_query_suffix_enabled = _env_enabled(
        "LMCACHE_APPEND_PASS_QUESTION_MARKS",
    )
    logger.info(
        "Phase query question-mark suffix enabled: %s",
        phase_query_suffix_enabled,
    )

    # --- Main Execution ---
    with builder(
        cfg.model_path,
        cfg.tp_size,
        bs=max_num_reqs,
        enable_holes=hole_enabled,
    ) as llm:
        with torch.no_grad():
            if cfg.method in ("cacheblend", "reuse") and hole_mode == "hole" and False:
                _run_hole_mode_preflight(
                    llm=llm,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    preinst_ids=preinst_ids,
                    postinst_ids=postinst_ids,
                    prefix_prompt=prefix_prompt,
                )

            # 1. Batched Document Prefilling (Only for 'cacheblend')
            prefill_summary = None
            if cfg.method == "cacheblend" or cfg.method == "reuse":
                prefill_summary = prefill_chunks(
                    llm,
                    eval_dataset,
                    cfg,
                    tokenizer,
                    preinst_ids,
                    prefix_prompt,
                    hole_mode=hole_mode,
                    hole_ids=hole_ids,
                    hole_positions=hole_positions,
                    phase1_submit_mode=phase1_submit_mode,
                    build_qa_prompt_fn=build_qa_prompt,
                )

            # 2. Phase 2 repeated runs:
            # pass-1 fills missing hole chunks, pass-2 measures pure reuse.
            phase_outputs: list[tuple[str, list]] = []
            for phase_idx in range(phase2_repeats):
                phase_name = f"phase2_pass{phase_idx + 1}"
                # Make each pass query text unique so query-side cache keys
                # are not reused across passes.
                pass_query_prompt = query_prompt
                if phase_query_suffix_enabled:
                    pass_query_prompt += "?" * (phase_idx + 1)
                # Keep LMCache state; clear only the local vLLM prefix cache.
                llm.reset_prefix_cache(
                    reset_running_requests=True, reset_connector=False
                )
                phase_output = run_batched_queries(
                    llm,
                    eval_dataset,
                    cfg,
                    tokenizer,
                    preinst_ids,
                    postinst_ids,
                    prefix_prompt,
                    pass_query_prompt,
                    build_qa_prompt_fn=build_qa_prompt,
                    compute_f1_fn=compute_f1,
                    compute_rl_fn=compute_rl,
                    phase2_submit_mode=phase2_submit_mode,
                    phase_label=phase_name,
                )
                phase_outputs.append((phase_name, phase_output))

            # 3. Dump Results
            if phase_outputs:
                # Backward-compatible default output: keep final pass in res.json.
                dump_res(phase_outputs[-1][1])
                output_dir = HydraConfig.get().run.dir
                phase_dump_path = os.path.join(output_dir, "res_phase2_passes.json")
                with open(phase_dump_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {phase_name: rows for phase_name, rows in phase_outputs},
                        f,
                        indent=2,
                    )
            if phase_outputs and phase_outputs[-1][1]:
                # Keep answer_quality.tsv schema unchanged; write final pass by default.
                dump_answer_quality(phase_outputs[-1][1])

            plot_path = plot_query_latency_hit_rate(phase_outputs)

            print("---------------Result Summary---------------------")
            print(f"phase1_submit_mode={phase1_submit_mode}")
            print(f"phase2_submit_mode={phase2_submit_mode}")
            if prefill_summary is not None:
                print("[phase1_prefill] --- Prefill Metrics (Seconds) ---")
                print(
                    f"[phase1_prefill] Average KV-cache prefill: "
                    f"{_fmt_metric(prefill_summary.get('avg_prefill_kv_cache_s'))} | "
                    f"Total prefill time: "
                    f"{_fmt_metric(prefill_summary.get('total_prefill_time_s'))} | "
                    "Num KV caches: "
                    f"{int(prefill_summary.get('num_prefill_kv_caches', 0))}"
                )
            for phase_name, phase_output in phase_outputs:
                if not phase_output:
                    print(f"[{phase_name}] no results")
                    continue
                summary = phase_output[-1]
                print(f"[{phase_name}] {summary.get('mean_f1', 0)=}")
                print(f"[{phase_name}] {summary.get('mean_rl', 0)=}")
                hit_rate = summary.get("cache_hit_rate", None)
                hit_rate_str = (
                    f"{float(hit_rate):.4f}" if hit_rate is not None else "N/A"
                )
                print(f"[{phase_name}] Average hit rate = {hit_rate_str}")
                print(f"[{phase_name}] --- Latency Metrics (Seconds) ---")
                print(
                    f"[{phase_name}] Average E2E: "
                    f"{_fmt_metric(summary.get('avg_e2e_s'))} | "
                    f"P90 E2E: {_fmt_metric(summary.get('p90_e2e_s'))}"
                )
                print(
                    f"[{phase_name}] Average TTFT: "
                    f"{_fmt_metric(summary.get('avg_ttft_s'))} | "
                    f"P90 TTFT: {_fmt_metric(summary.get('p90_ttft_s'))}"
                )
                print(
                    f"[{phase_name}] Average TBT: "
                    f"{_fmt_metric(summary.get('avg_tbt_s'))} | "
                    f"P90 TBT: {_fmt_metric(summary.get('p90_tbt_s'))}"
                )

            if plot_path is not None:
                print(f"Per-query latency/hit-rate plot: {plot_path}")


if __name__ == "__main__":
    main()
