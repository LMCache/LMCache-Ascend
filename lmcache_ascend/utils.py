# Adapted from https://github.com/YaoJiayi/CacheBlend/blob/main/example/utils.py
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig, AutoModelForCausalLM
import json
import torch
import collections
import string
import re
from rouge_score import rouge_scorer
from typing import Any
from transformers import AutoTokenizer
import gc
import subprocess
from vllm import LLM
import os
import pathlib
import random
import string
import sys
import shutil

def read_all_tensors(path):
    d = {}
    for pt_file in pathlib.Path(path).glob("*.pt"):
        data: torch.Tensor = torch.load(pt_file)
        d[pt_file.name] = data.tolist()
    return d

def parse_npu_process_info(npu_id, chip_id):
    try:
        output = subprocess.check_output(['npu-smi', 'info'], text=True)
        pattern = re.compile( r'\|\s*(\d+)\s+(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|')

        # Loop through each line and apply regex
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                npu, chip = int(match.group(1)), int(match.group(2))
                if npu == npu_id and chip == chip_id:
                    process_id = match.group(3).strip()
                    process_name = match.group(4).strip()
                    mem_mb = match.group(5).strip()
                    return {
                            "npu": npu,
                            "chip": chip,
                            "pid": process_id,
                            "name": process_name,
                            "memory_mb": int(mem_mb)
                            }

        return None  # Not found
    except subprocess.CalledProcessError as e:
        print("Error running npu-smi:", e)
        return None

def get_mem(npu_id, chip_id=0):
    try:
        return parse_npu_process_info(npu_id,chip_id)['memory_mb']
    except:
        return -1

def report_npu_tensors2():
    import gc
    import torch

    counter = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'npu':
                key = (tuple(obj.shape), str(obj.dtype))
                counter[key] = counter.get(key, 0) + 1
        except Exception:
            pass

    print("NPU tensor summary:")
    total_bytes = 0
    for (shape, dtype), count in counter.items():
        try:
            # Get element size
            element_size = torch.empty((), dtype=getattr(torch, dtype.split('.')[1])).element_size()
            # Total number of elements in one tensor
            numel = torch.tensor(shape).prod().item()
            # Total size in bytes
            size_bytes = count * numel * element_size
            size_mb = size_bytes / 1e6
            total_bytes += size_bytes
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → {size_mb:.2f} MB total")
        except Exception:
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → [Size calc failed]")

    print(f"~{total_bytes / 1e6:.2f} MB total on NPU")

def docs_to_ids(docs: list[str], tokenizer: AutoTokenizer) -> list[int]:
    res = []
    if tokenizer.bos_token_id is not None:
        res.append(tokenizer.bos_token_id)

    for doc in docs:
        # Tokenize and convert to input IDs
        input_ids = tokenizer.encode(doc, add_special_tokens=False)
        res.extend(input_ids)

    return res

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s: str):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def normalize_answer(s: str):
    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _extract_contexts(example: dict[str, Any]) -> list[Any]:
    if "ctxs" in example:
        contexts = example["ctxs"]
    elif "context" in example:
        contexts = example["context"]
    else:
        contexts = []
    if not isinstance(contexts, list):
        raise TypeError(f"Expected list for contexts, got {type(contexts)}")
    return contexts


def _format_context(ctx: Any) -> str:
    # Most common case for 2wiki-style files.
    if isinstance(ctx, str):
        return ctx

    # Hotpot/musique-like entries.
    if isinstance(ctx, dict):
        title = str(ctx.get("title", "")).strip()
        text = str(ctx.get("text", "")).strip()
        if title and text:
            return f"{title}\n\n{text}\n\n"
        if text:
            return text
        if title:
            return title
        return str(ctx)

    # Some datasets encode context as [title, text] or (title, text).
    if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
        title = str(ctx[0]).strip()
        text = str(ctx[1]).strip()
        if title and text:
            return f"{title}\n\n{text}\n\n"
        return text if text else title

    return str(ctx)


def build_qa_prompt(example: dict[str, Any], query_prompt: str):
    """
    example['question']: str
    example['ctxs']: list[str] the documents
    """
    q = normalize_question(example["question"])
    doc_prompts = [_format_context(ctx) for ctx in _extract_contexts(example)]
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def compute_f1(pred: str, gold: str|list[str]|list[list[str]], tokenizer: PreTrainedTokenizerBase):
    if not isinstance(gold, str):
        res = 0.0
        for a in gold:
            res = max(res, compute_f1(pred, a, tokenizer))
        return res
    pred = parse_generation(pred)
    gold = parse_generation(gold)
    gold_toks = tokenizer.encode(normalize_answer(gold), add_special_tokens=False)
    pred_toks = tokenizer.encode(normalize_answer(pred), add_special_tokens=False)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred: str, gold: str|list[str]|list[list[str]]):
    if not isinstance(gold, str):
        res = 0.0
        for a in gold:
            res = max(res, compute_rl(pred, a))
        return res
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

metric_name2f = {
    'f1': compute_f1,
    'rl': compute_rl
}
