# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional, Union, no_type_check
import argparse
import contextlib
import logging
import random
import time

# Third Party
from lmcache.config import LMCacheEngineConfig as Config
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.v1.config import LMCacheEngineConfig as V1Config
from transformers import AutoConfig, AutoTokenizer
from utils import (
    PromptBuildMethodType,
    build_fewshot_prompt,
    build_qa_prompt,
    build_rag_prompt_tokens,
    compute_f1,
    compute_rl,
    init_logger,
    load_dataset,
)
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
import openai
import pandas as pd

logger = init_logger(__name__, logging.INFO)

SYSTEM_PROMPT_SET = {
    PromptBuildMethodType.QA: (
        "You will be asked a question after reading several passages. "
        "Please directly answer the question based on the given passages. "
        "Do NOT repeat the question. "
        "The answer should be within 5 words.\nPassages:\n"
    ),
    PromptBuildMethodType.FEW_SHOT: (
        "Summarize the dialogue into a few short sentences. "
        " The following are some examples.\n\n"
    ),
}
QUERY_PROMPT_SET = {
    PromptBuildMethodType.QA: (
        "\n\nAnswer the question directly based on the given passages."
        " Do NOT repeat the question. "
        "The answer should be within 5 words. \nQuestion:"
    ),
    PromptBuildMethodType.FEW_SHOT: "",
}


@contextlib.contextmanager
def build_llm_with_lmcache(model: str, max_model_len: int = 32000, blend: bool = True):
    """Build LLM with LMCache for offline serving"""

    LMCACHE_CONNECTOR = "LMCacheAscendConnectorV1Dynamic"
    CONNECTOR_PATH = "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"

    ktc = KVTransferConfig(
        kv_connector=LMCACHE_CONNECTOR,
        kv_role="kv_both",
        kv_connector_module_path=CONNECTOR_PATH,
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=(False if blend else True),
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        pass


@dataclass
class WorkloadConfig:
    """Configuration for a single RAG workload."""

    # Model name
    model: str
    # Tokenizer name
    tokenizer: str
    # Dataset.
    dataset: str
    # Start index of the workload
    start_index: int
    # End index of the workload
    end_index: int
    # Random shuffle.
    shuffle: bool
    # System prompt.
    system_prompt: str
    # Separator.
    separator: str
    # Query prompt.
    query_prompt: str
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # Max tokens for each generation.
    max_tokens: int
    # KV chunk size
    kv_chunk_size: int
    # LMCache config
    lmconfig: Union[Config, V1Config]
    # Online specific configs
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    temperature: float = 0.0


@dataclass
class Response:
    request_id: int
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float
    # Additional metrics for online serving
    tpot: float = 0.0  # Time per output token
    end_to_end_latency: float = 0.0


def parse_arguments():
    """Parse RAG benchmark configurations."""
    parser = argparse.ArgumentParser(description="Parse RAG benchmark configurations.")
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
    )
    # Optional/Default arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer name (defaults to model if empty)",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument("--shuffle", action="store_true", help="Random shuffle")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--query-prompt", type=str, default="", help="Query prompt")
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name for the summary csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens for each generation",
    )
    # Online mode arguments
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use online serving mode with OpenAI-compatible API",
    )
    parser.add_argument(
        "--openai-api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI API base URL for online mode",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="dummy-key",
        help="OpenAI API key for online mode",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )

    # LMCache specific argument
    # (if needed, although usually handled by lmcache_get_or_create_config)
    parser.add_argument(
        "--kv-chunk-size", type=int, default=256, help="KV chunk size for LMCache"
    )

    args = parser.parse_args()
    return args


def parse_size(size: str) -> int:
    """Parse size string like '30GB' to bytes"""
    if not size:
        return -1

    size = size.upper()
    unit_multipliers = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "B": 1,
    }

    for unit, multiplier in unit_multipliers.items():
        if size.endswith(unit):
            try:
                return int(size[: -len(unit)]) * multiplier
            except ValueError as ve:
                raise ValueError(f"Invalid size value in {size}") from ve

    raise ValueError(f"Invalid size unit {size}")


class KVSizeCalculator:
    """Calculates KV cache size based on model configuration."""

    def __init__(
        self,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        precision: int,
    ):
        # ratio = heads * dim * layers * precision * (k + v)
        self.ratio = num_key_value_heads * head_dim * num_layers * precision * 2
        logger.info(
            f"num_key_value_heads:{num_key_value_heads} head_dim:{head_dim} "
            f"num_layers:{num_layers} precision:{precision}"
        )

    def get_kv_size(self, token_cnt: int) -> int:
        """Returns the KV size in bytes for a given number of tokens."""
        return token_cnt * self.ratio


class BaseRAGManager(ABC):
    """Abstract base class for RAG managers to improve code reuse"""

    def __init__(self, workload_config: WorkloadConfig):
        self.workload_config = workload_config
        self._tokenizer = AutoTokenizer.from_pretrained(workload_config.tokenizer)
        self._model_config = AutoConfig.from_pretrained(workload_config.model)
        self._answers = []
        self._build_method = workload_config.prompt_build_method
        self._results = []

        # Load and preprocess dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load and preprocess dataset - common logic for both online and offline"""
        eval_dataset = load_dataset(self.workload_config.dataset)
        start_index = self.workload_config.start_index
        end_index = self.workload_config.end_index

        if end_index < 0 or end_index > len(eval_dataset):
            end_index = len(eval_dataset)

        eval_dataset = eval_dataset[start_index:end_index]

        if self.workload_config.shuffle:
            random.shuffle(eval_dataset)

        self._eval_dataset = eval_dataset
        self._answers = [ex["answers"] for ex in eval_dataset]

    @abstractmethod
    def _precompute_documents(self, *args, **kwargs) -> int:
        """Abstract method for precomputing document KV cache/prompts"""
        pass

    @abstractmethod
    def run_benchmark(self, **kwargs) -> float:
        """Abstract method for running the benchmark"""
        pass

    def summary(self, total_time: float, is_online: bool = False) -> pd.DataFrame:
        """Generate summary statistics - common logic with online-specific metrics"""
        cnt = len(self._results)
        assert cnt > 0, "No results to summarize"

        generation_times = [r.generation_time for r in self._results]
        prefill_token_cnts = [r.prompt_tokens for r in self._results]
        generation_token_cnts = [r.generation_tokens for r in self._results]

        quality = []
        for i in range(cnt):
            generated_text = self._results[i].body
            expected_answers = self._answers[i]

            if self._build_method == PromptBuildMethodType.QA:
                quality_score = max(
                    compute_f1(generated_text, answer, self._tokenizer)
                    for answer in expected_answers
                )
            elif self._build_method == PromptBuildMethodType.FEW_SHOT:
                quality_score = max(
                    compute_rl(generated_text, answer) for answer in expected_answers
                )
            else:
                raise ValueError(f"Invalid prompt build method {self._build_method}")
            quality.append(quality_score)

        avg_quality = sum(quality) / cnt
        thput = cnt / total_time

        summary_data = {
            "quality": quality,
            "generation_time": generation_times,
            "prefill_token_cnt": prefill_token_cnts,
            "generation_token_cnt": generation_token_cnts,
        }

        default_cols = [
            "quality",
            "generation_time",
            "prefill_token_cnt",
            "generation_token_cnt",
        ]

        if is_online:
            ttfts = [r.ttft for r in self._results]
            end_to_end_latencies = [r.end_to_end_latency for r in self._results]
            tpots = [r.tpot for r in self._results]

            avg_ttft = sum(ttfts) / cnt
            avg_tpot = sum(tpots) / cnt
            avg_e2e_latency = sum(end_to_end_latencies) / cnt

            summary_data.update(
                {
                    "ttft": ttfts,
                    "tpot": tpots,
                    "end_to_end_latency": end_to_end_latencies,
                }
            )

            final_cols_order = [
                "quality",
                "ttft",
                "tpot",
                "end_to_end_latency",
                "generation_time",
                "prefill_token_cnt",
                "generation_token_cnt",
            ]

            df = pd.DataFrame(summary_data)
            df = df[final_cols_order]

            logger.info(
                f"Summary: {cnt} requests, average_ttft={avg_ttft:.4f} (second)\n"
                f"average_tpot={avg_tpot:.4f} (second)\n"
                f"average_e2e_latency={avg_e2e_latency:.4f} (second)\n"
                f"throughput={thput:.4f} (req/s)\n"
                f"average_quality={avg_quality:.4f}\n"
            )
        else:
            summary_data["throughput"] = [thput] * cnt
            df = pd.DataFrame(summary_data)

            offline_cols_order = default_cols + ["throughput"]
            df = df[offline_cols_order]

            logger.info(
                f"Summary: {cnt} requests, total_time={total_time:.4f} (second)\n"
                f"throughput={thput:.4f} (req/s)\n"
                f"average_quality={avg_quality:.4f}\n"
            )

        return df


class OfflineRAGManager(BaseRAGManager):
    """Manages RAG benchmark for offline serving using vLLM."""

    def __init__(self, workload_config: WorkloadConfig):
        super().__init__(workload_config)

        self._document_tokens = []  # Store document tokens for precompute
        self._request_tokens = []  # Store full request tokens

        # Preprocess all prompts into token format
        self._prepare_tokens()

    def _encode_prompt(self, prompt: str, add_special_tokens: bool = True) -> List[int]:
        """Helper to encode prompt, using tokenizer from base class."""
        return self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def _prepare_tokens(self):
        """Prepare all prompts and documents into token lists."""
        config = self.workload_config
        system_prompt_tokens = self._encode_prompt(config.system_prompt)
        query_prompt_tokens = self._encode_prompt(
            config.query_prompt, add_special_tokens=False
        )
        separator_tokens = self._encode_prompt(
            config.separator, add_special_tokens=False
        )

        for ex in self._eval_dataset:
            if config.prompt_build_method == PromptBuildMethodType.QA:
                doc_prompts, q_prompt = build_qa_prompt(ex, config.query_prompt)
            elif config.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
                doc_prompts, q_prompt = build_fewshot_prompt(ex)
            else:
                raise ValueError(
                    f"Invalid prompt build method {config.prompt_build_method}"
                )

            doc_tokens_list = [
                self._encode_prompt(doc, add_special_tokens=False)
                for doc in doc_prompts
                if doc
            ]
            full_q_tokens = self._encode_prompt(q_prompt, add_special_tokens=False)

            prompt_tokens = build_rag_prompt_tokens(
                system_prompt_tokens,
                doc_tokens_list,
                full_q_tokens,
                separator_tokens,
            )
            self._request_tokens.append(prompt_tokens)

            random.shuffle(doc_tokens_list)

            fix_doc_tokens_list = []
            if config.lmconfig.enable_blending:
                for doc_tokens in doc_tokens_list:
                    fix_doc_tokens_list.append(
                        system_prompt_tokens
                        + separator_tokens
                        + doc_tokens
                        + separator_tokens
                        + query_prompt_tokens
                    )

            self._document_tokens.append(fix_doc_tokens_list)

            if not fix_doc_tokens_list and config.lmconfig.enable_blending:
                logger.info(
                    "Blending enabled but no valid documents found for this example."
                )

    def _precompute_documents(self, llm: LLM) -> int:
        """Precompute KV cache for document chunks using the same LLM instance"""
        if not self.workload_config.lmconfig.enable_blending:
            logger.info("LMCache blending disabled. Skipping document precomputation.")
            return 0

        logger.info("Starting document precomputation...")

        model_config = self._model_config
        head_dim = (
            model_config.head_dim
            if model_config.head_dim is not None
            else model_config.hidden_size // model_config.num_attention_heads
        )

        kv_size_calculator = KVSizeCalculator(
            model_config.num_key_value_heads,
            head_dim,
            model_config.num_hidden_layers,
            2,  # FP16 precision
        )

        current_size_taken = 0
        precomputed_count = 0
        round_up_token_cnt = self.workload_config.kv_chunk_size

        kv_storage_size_gb = self.workload_config.lmconfig.max_local_cpu_size
        GB_TO_BYTES = 1024 * 1024 * 1024
        kv_storage_size_bytes = int(kv_storage_size_gb * GB_TO_BYTES)

        logger.info(
            f"KV cache storage size limit: {kv_storage_size_gb} GB"
            " ({kv_storage_size_bytes} bytes)"
        )

        for i, doc_tokens_list in enumerate(self._document_tokens):
            if current_size_taken >= kv_storage_size_bytes:
                logger.info(
                    f"KV cache size limit reached ({kv_storage_size_gb} GB)."
                    " Stopping precomputation."
                )
                break

            # Calculate size for this document set
            total_doc_tokens = sum(len(doc_tokens) for doc_tokens in doc_tokens_list)
            # Round up to chunk size
            total_doc_tokens = (
                (total_doc_tokens + round_up_token_cnt - 1) // round_up_token_cnt
            ) * round_up_token_cnt

            this_case_size = kv_size_calculator.get_kv_size(total_doc_tokens)

            if current_size_taken + this_case_size > kv_storage_size_bytes:
                logger.info(
                    f"Document set {i} requires {this_case_size} bytes, "
                    f"which exceeds the remaining limit. Stopping precomputation."
                )
                break

            # Precompute each document chunk
            for doc_tokens in doc_tokens_list:
                # Use minimal generation to trigger KV cache storage
                sampling_params = SamplingParams(temperature=0, max_tokens=1)
                try:
                    llm.generate(
                        prompt_token_ids=doc_tokens, sampling_params=sampling_params
                    )
                except Exception as e:
                    logger.warning(f"Precompute failed for document chunk: {e}")
                    continue

            current_size_taken += this_case_size
            precomputed_count += 1

        logger.info(
            f"Precomputed {precomputed_count} document sets, "
            "used {current_size_taken} bytes of KV cache"
        )
        return precomputed_count

    @no_type_check
    def run_benchmark(
        self, llm: LLM, sampling_params: SamplingParams, **kwargs
    ) -> float:
        """Run the benchmark - optimized using vLLM's batch processing approach"""
        self._results = []

        precomputed_count = self._precompute_documents(llm)
        logger.info(f"Precomputed {precomputed_count} document chunks for KV cache")

        logger.info("Starting benchmark (throughput-focused with batch processing)...")

        prompts = [
            TokensPrompt(prompt_token_ids=tokens) for tokens in self._request_tokens
        ]
        sampling_params_list = [sampling_params] * len(self._request_tokens)

        elapsed_time = 0.0

        try:
            start_time = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params_list)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            avg_generation_time = elapsed_time / len(outputs) if outputs else 0.0
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                prompt_token_count = len(output.prompt_token_ids)
                generation_token_count = len(output.outputs[0].token_ids)

                response = Response(
                    request_id=i,
                    body=generated_text,
                    ttft=0.0,
                    generation_time=avg_generation_time,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=generation_token_count,
                    launch_time=start_time,
                    finish_time=end_time,
                )
                self._results.append(response)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            elapsed_time = self._run_sequential_fallback(
                llm, sampling_params, self._request_tokens
            )

        if self._results:
            total_prompt_tokens = sum(r.prompt_tokens for r in self._results)
            total_output_tokens = sum(r.generation_tokens for r in self._results)
            total_tokens = total_prompt_tokens + total_output_tokens

            print("\n=== Throughput Results ===")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Total requests: {len(self._results)}")
            print(f"Total prompt tokens: {total_prompt_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Total tokens: {total_tokens}")
            print(f"Requests per second: {len(self._results) / elapsed_time:.2f}")
            print(f"Tokens per second: {total_tokens / elapsed_time:.2f}")
            print(f"Output tokens per second: {total_output_tokens / elapsed_time:.2f}")

        logger.info(f"Completed {len(self._results)} requests")
        return elapsed_time

    def _run_sequential_fallback(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        prompt_tokens_list: List[List[int]],
    ) -> float:
        """Fallback to sequential processing if batch processing fails"""
        self._results = []

        start_time = time.perf_counter()

        individual_request_times = []

        for i, prompt_tokens in enumerate(prompt_tokens_list):
            request_start_time = time.perf_counter()
            try:
                output = llm.generate(
                    prompt_token_ids=[prompt_tokens], sampling_params=sampling_params
                )
                request_end_time = time.perf_counter()

                generated_text = output[0].outputs[0].text
                prompt_token_count = len(output[0].prompt_token_ids)
                generation_token_count = len(output[0].outputs[0].token_ids)

                request_duration = request_end_time - request_start_time

                response = Response(
                    request_id=i,
                    body=generated_text,
                    ttft=0.0,
                    generation_time=request_duration,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=generation_token_count,
                    launch_time=request_start_time,
                    finish_time=request_end_time,
                )
                self._results.append(response)
                individual_request_times.append(request_duration)

            except Exception as e:
                request_end_time = time.perf_counter()
                logger.error(f"Error processing request {i} sequentially: {e}")

                # Add dummy response for failed requests
                request_duration = request_end_time - request_start_time
                response = Response(
                    request_id=i,
                    body="",
                    ttft=0.0,
                    generation_time=request_duration,
                    prompt_tokens=len(prompt_tokens),
                    generation_tokens=0,
                    launch_time=request_start_time,
                    finish_time=request_end_time,
                )
                self._results.append(response)
                individual_request_times.append(request_duration)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        return elapsed_time


class OnlineRAGManager(BaseRAGManager):
    """
    Online RAG Manager that uses OpenAI-compatible API
    instead of local vLLM instance
    """

    def __init__(self, workload_config: WorkloadConfig):
        super().__init__(workload_config)
        # Initialize OpenAI client
        self._client = openai.OpenAI(
            base_url=workload_config.openai_api_base,
            api_key=workload_config.openai_api_key or "dummy-key",
        )

        self._document_prompts = []  # Store document prompts as strings
        self._request_prompts = []  # Store full request prompts as strings

        # Process dataset for online serving (using strings, not tokens)
        self._prepare_prompts()

    def _prepare_prompts(self):
        """Prepare prompts as strings for online API calls"""
        config = self.workload_config
        system_prompt = config.system_prompt
        separator = config.separator
        query_prompt = config.query_prompt

        for ex in self._eval_dataset:
            if config.prompt_build_method == PromptBuildMethodType.QA:
                doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
            elif config.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
                doc_prompts, q_prompt = build_fewshot_prompt(ex)
            else:
                raise ValueError(
                    f"Invalid prompt build method {config.prompt_build_method}"
                )

            doc_prompts_for_precompute = []
            if config.lmconfig.enable_blending and doc_prompts:
                doc_prompts_for_precompute.append(
                    separator.join([system_prompt] + doc_prompts + [query_prompt])
                )

            full_prompt = separator.join([system_prompt] + doc_prompts + [q_prompt])
            self._request_prompts.append(full_prompt)

            if not doc_prompts_for_precompute:
                if config.lmconfig.enable_blending:
                    logger.info(
                        "Blending enabled but no documents for precomputation"
                        " in this example."
                    )
                self._document_prompts.append(None)
            else:
                self._document_prompts.append(doc_prompts_for_precompute)

    def _send_warmup_request(self):
        """Send a warmup request to initialize the model/cache on the server"""
        logger.info("Sending warmup requests...")
        warmup_prompt = "Hello, this is a warmup request."

        try:
            self._client.chat.completions.create(
                model=self.workload_config.model,
                messages=[{"role": "user", "content": warmup_prompt}],
                temperature=1,
                max_tokens=1,
                timeout=20,
            )
            logger.info("Warmup request completed successfully.")
            return True

        except Exception as e:
            logger.warning(f"Warmup request failed: {e}")
            return False

    def _precompute_documents(self) -> int:
        """Precompute documents by sending them to the online server"""
        if not self.workload_config.lmconfig.enable_blending:
            logger.info(
                "LMCache blending disabled. Skipping online document precomputation."
            )
            return 0

        logger.info("Starting online document precomputation...")

        precomputed_count = 0

        all_doc_prompts: List[str] = []
        for doc_set in self._document_prompts:
            if doc_set:
                all_doc_prompts.extend(doc_set)

        if not all_doc_prompts:
            logger.info("No documents to precompute.")
            return 0

        for i, doc_prompt in enumerate(all_doc_prompts):
            try:
                self._client.chat.completions.create(
                    model=self.workload_config.model,
                    messages=[{"role": "user", "content": doc_prompt}],
                    temperature=self.workload_config.temperature,
                    max_tokens=1,  # Minimal generation
                    timeout=30,
                )
                precomputed_count += 1

            except Exception as e:
                logger.warning(f"Precompute failed for document chunk {i}: {e}")
                continue

        logger.info(
            f"Precomputed {precomputed_count} document chunks for online serving"
        )
        return precomputed_count

    def run_benchmark(self, **kwargs) -> float:
        """Run online benchmark with detailed timing metrics"""
        self._results = []

        # Step 1: Precompute documents
        if self.workload_config.lmconfig.enable_blending:
            # Use original precomputation logic for cacheblend
            precomputed_count = self._precompute_documents()
            logger.info(f"Precomputed {precomputed_count} document sets")
        else:
            # Send warmup request when cacheblend is disabled
            self._send_warmup_request()

        # Step 2: Run benchmark requests with detailed timing
        logger.info("Starting online benchmark...")

        start_time = time.perf_counter()

        for i, request_prompt in enumerate(self._request_prompts):
            request_start_time = time.perf_counter()
            ttft_time = 0.0  # Time to First Token
            generated_text = ""
            completion_tokens = 0

            # --- Stream API Call Section ---
            try:
                prompt_tokens = (
                    len(self._tokenizer.encode(request_prompt))
                    if hasattr(self, "_tokenizer")
                    else 0
                )
                # Make API call with timing
                stream = self._client.chat.completions.create(
                    model=self.workload_config.model,
                    messages=[{"role": "user", "content": request_prompt}],
                    temperature=self.workload_config.temperature,
                    max_tokens=self.workload_config.max_tokens,
                    stream=True,
                    timeout=60,
                )

                first_token_received = False
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        if not first_token_received:
                            ttft_time = time.perf_counter() - request_start_time
                            first_token_received = True

                        generated_text += content
                        completion_tokens += 1

                request_end_time = time.perf_counter()
                # --- End Stream API Call Section ---

                # Extract response details
                if (
                    completion_tokens == 0
                    and not generated_text
                    and first_token_received
                ):
                    completion_tokens = 1

                # Calculate timing metrics
                total_time = request_end_time - request_start_time

                if completion_tokens > 1:
                    tpot = (total_time - ttft_time) / (completion_tokens - 1)
                elif completion_tokens == 1:
                    tpot = 0.0
                else:
                    tpot = 0.0

                logger.info(f"completion_tokens:{completion_tokens}")

                response = Response(
                    request_id=i,
                    body=generated_text or "",
                    ttft=ttft_time,
                    generation_time=total_time,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=completion_tokens,
                    launch_time=request_start_time,
                    finish_time=request_end_time,
                    tpot=tpot,
                    end_to_end_latency=total_time,
                )

                self._results.append(response)

            except Exception as e:
                logger.error(f"Error processing request {i}: {e}")
                # Add dummy response for failed requests
                request_end_time = time.perf_counter()

                response = Response(
                    request_id=i,
                    body="",
                    ttft=0.0,
                    generation_time=0.0,
                    prompt_tokens=len(self._tokenizer.encode(request_prompt)),
                    generation_tokens=0,
                    launch_time=request_start_time,
                    finish_time=request_end_time,
                    tpot=0.0,
                    end_to_end_latency=0.0,
                )
                self._results.append(response)

        end_time = time.perf_counter()
        total_elapsed = end_time - start_time

        # Print throughput metrics
        total_prompt_tokens = sum(r.prompt_tokens for r in self._results)
        total_output_tokens = sum(r.generation_tokens for r in self._results)
        successful_requests = len([r for r in self._results if r.generation_tokens > 0])

        print("\n=== Online Serving Results ===")
        print(f"Elapsed time: {total_elapsed:.2f} seconds")
        total_prompt_tokens = sum(r.prompt_tokens for r in self._results)
        total_output_tokens = sum(r.generation_tokens for r in self._results)
        successful_requests = len([r for r in self._results if r.generation_tokens > 0])
        successful_results = [r for r in self._results if r.generation_tokens > 0]
        num_successful = len(successful_results)

        if num_successful > 0:
            avg_ttft = sum(r.ttft for r in successful_results) / num_successful
            avg_tpot = sum(r.tpot for r in successful_results) / num_successful
            avg_e2e_latency = (
                sum(r.end_to_end_latency for r in successful_results) / num_successful
            )
        else:
            avg_ttft, avg_tpot, avg_e2e_latency = 0.0, 0.0, 0.0

        print(f"Total requests: {len(self._results)}")
        print(f"Successful requests: {successful_requests}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Requests per second: {len(self._results) / total_elapsed:.2f}")
        if total_output_tokens > 0:
            print(
                f"Output tokens per second: {total_output_tokens / total_elapsed:.2f}"
            )

        print(f"Average TTFT (Successful): {avg_ttft:.4f} seconds")
        print(f"Average TPOT (Successful): {avg_tpot:.4f} seconds")
        print(f"Average End-to-End Latency (Successful): {avg_e2e_latency:.4f} seconds")

        logger.info(f"Completed {len(self._results)} requests")
        return total_elapsed


def run_rag_benchmark(args):
    """Main function to run the RAG benchmark"""
    build_prompt_method_str = args.prompt_build_method.upper()
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")

    lmconfig = lmcache_get_or_create_config()
    if args.online:
        # Online mode configuration
        workload_config = WorkloadConfig(
            model=args.model,
            tokenizer=args.tokenizer,
            dataset=args.dataset,
            start_index=args.start_index,
            end_index=args.end_index,
            shuffle=args.shuffle,
            system_prompt=args.system_prompt,
            separator=lmconfig.blend_special_str,  # Simple separator for online mode
            query_prompt=args.query_prompt,
            prompt_build_method=build_prompt_method,
            max_tokens=args.max_tokens,
            kv_chunk_size=lmconfig.chunk_size,  # Default chunk size for online mode
            lmconfig=lmconfig,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
            temperature=args.temperature,
        )

        manager = OnlineRAGManager(workload_config)
        total_time = manager.run_benchmark()

        logger.info(f"Finished online benchmarking, dumping summary to {args.output}")
        summary = manager.summary(total_time, is_online=True)
        summary.to_csv(args.output, index=False)
    else:
        # Offline mode configuration (existing logic)
        workload_config = WorkloadConfig(
            model=args.model,
            tokenizer=args.tokenizer,
            dataset=args.dataset,
            start_index=args.start_index,
            end_index=args.end_index,
            shuffle=args.shuffle,
            system_prompt=args.system_prompt,
            separator=lmconfig.blend_special_str,
            query_prompt=args.query_prompt,
            prompt_build_method=build_prompt_method,
            max_tokens=args.max_tokens,
            kv_chunk_size=lmconfig.chunk_size,
            lmconfig=lmconfig,
        )

        manager = OfflineRAGManager(workload_config)

        with build_llm_with_lmcache(args.model, 32000, lmconfig.enable_blending) as llm:
            sampling_params = SamplingParams(
                temperature=args.temperature, max_tokens=args.max_tokens
            )

            total_time = manager.run_benchmark(llm, sampling_params)

            logger.info(
                f"Finished offline benchmarking, dumping summary to {args.output}"
            )
            summary = manager.summary(total_time, is_online=False)
            summary.to_csv(args.output, index=False)


def main():
    args = parse_arguments()

    build_prompt_method_str = args.prompt_build_method.upper()
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")

    if len(args.system_prompt) == 0:
        args.system_prompt = SYSTEM_PROMPT_SET[build_prompt_method]
    if len(args.query_prompt) == 0:
        args.query_prompt = QUERY_PROMPT_SET[build_prompt_method]
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model

    args.system_prompt = args.system_prompt.encode().decode("unicode_escape")
    args.query_prompt = args.query_prompt.encode().decode("unicode_escape")

    if args.verbose:
        global logger
        logger = init_logger(__name__, log_level=logging.DEBUG)

    run_rag_benchmark(args)


if __name__ == "__main__":
    main()
