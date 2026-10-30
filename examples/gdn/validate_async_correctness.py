# SPDX-License-Identifier: Apache-2.0
"""Compare sync/async long-context answers after verified CPU cache restores."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import urllib.request

METRICS = (
    "prefix_cache_hits_total",
    "external_prefix_cache_hits_total",
    "num_preemptions_total",
    "num_requests_running",
    "num_requests_waiting",
)
WORDS = (
    "river stone garden forest paper window silver orange valley cloud table road"
).split()


class Client:
    """HTTP client for one dedicated test server; never controls its processes."""

    def __init__(self, base_url: str, model: str, timeout: float):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base-url must be an HTTP(S) server URL")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def request(self, path: str, body: dict | None = None, *, post=False):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode() if body is not None else None,
            method="POST" if post or body is not None else "GET",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode()
        return raw if path == "/metrics" else json.loads(raw) if raw else None

    def messages(self, prefix: str, question: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prefix + "\n" + question}]

    def token_count(self, prefix: str, question: str) -> int:
        response = self.request(
            "/tokenize",
            {
                "model": self.model,
                "messages": self.messages(prefix, question),
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return int(response["count"])

    def complete(self, prefix: str, question: str, max_tokens=128) -> dict:
        return self.request(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": self.messages(prefix, question),
                "temperature": 0,
                "seed": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )


def metrics(raw: str) -> dict[str, float]:
    """Require every cache/traffic counter used to verify the measured path."""
    values = {}
    for name in METRICS:
        matches = re.findall(
            r"^vllm:" + name + r"(?:\{[^}\n]*\})?\s+([0-9.eE+-]+)", raw, re.M
        )
        if not matches:
            raise ValueError("Missing required metric: " + name)
        value = sum(map(float, matches))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid metric: " + name)
        values[name] = value
    return values


def wait_idle(client: Client, timeout: float) -> dict[str, float]:
    deadline, stable = time.monotonic() + timeout, 0
    while True:
        values = metrics(client.request("/metrics"))
        busy = values["num_requests_running"] or values["num_requests_waiting"]
        stable = 0 if busy else stable + 1
        if stable >= 2:
            return values
        if time.monotonic() >= deadline:
            raise TimeoutError("Server is not idle; require a dedicated test server")
        time.sleep(0.2)


def check_control_api(spec: dict) -> None:
    """Reject reset APIs that cannot explicitly retain external CPU storage."""
    paths = spec.get("paths", {})
    for path in ("/pause", "/resume", "/reset_prefix_cache", "/tokenize"):
        if path not in paths:
            raise ValueError("Required development API is unavailable: " + path)

    def schema(value):
        while "$ref" in value:
            name = value["$ref"].split("/")[-1]
            value = spec["components"]["schemas"][name]
        return value

    def parameters(path):
        return {
            item["name"]: schema(item.get("schema", {}))
            for item in paths[path]["post"].get("parameters", [])
            if item.get("in") == "query"
        }

    pause, reset = parameters("/pause"), parameters("/reset_prefix_cache")
    if "wait" not in pause.get("mode", {}).get("enum", []):
        raise ValueError("The pause API must support mode=wait")
    for value in (pause.get("clear_cache", {}), reset.get("reset_external", {})):
        if value.get("type") != "boolean":
            raise ValueError("Cannot verify non-clearing pause/reset API")
    if reset.get("reset_running_requests", {}).get("type") != "boolean":
        raise ValueError("Cannot verify the non-aborting reset API")


def clear_npu_only(client: Client, timeout: float) -> None:
    """Drain inference, clear native prefix cache, and retain external CPU data."""
    wait_idle(client, timeout)
    client.request("/pause?mode=wait&clear_cache=false", post=True)
    try:
        result = client.request(
            "/reset_prefix_cache?reset_running_requests=false&reset_external=false",
            post=True,
        )
        if result is False:
            raise RuntimeError("Native prefix cache reset was refused")
    finally:
        client.request("/resume", post=True)
    wait_idle(client, timeout)


@dataclass
class Case:
    case_id: str
    prefix: str
    question: str
    answer: str
    prompt_tokens: int
    needle_end_tokens: int

    def digest(self) -> str:
        return hashlib.sha256((self.prefix + "\n" + self.question).encode()).hexdigest()


def make_case(client: Client, case_id: str, target: int, depth: float) -> Case:
    """Generate a reproducible passkey at a requested approximate context depth."""
    seed = int(hashlib.sha256(case_id.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    answer = f"{rng.getrandbits(64):016x}"
    words = [rng.choice(WORDS) for _ in range(target)]
    question = f"What is the secret code for record {case_id}? Return only the code."
    for _ in range(8):
        position = int(len(words) * depth)
        prefix = (
            f"Audit document {case_id}. Read all records carefully.\n<records>\n"
            + " ".join(words[:position])
            + f"\nRecord {case_id} has the secret code {answer}.\n"
            + " ".join(words[position:])
            + "\n</records>"
        )
        count = client.token_count(prefix, question)
        if target <= count <= target + 256:
            needle_end = prefix.index(answer) + len(answer)
            # Include template overhead as a conservative upper bound.
            restored_needle = client.token_count(prefix[:needle_end], "")
            return Case(case_id, prefix, question, answer, count, restored_needle)
        size = max(1, int(len(words) * target / count) + (8 if count < target else 0))
        words = (
            words[:size]
            if size <= len(words)
            else words + [rng.choice(WORDS) for _ in range(size - len(words))]
        )
    raise RuntimeError("Could not fit the requested context token budget")


def check_answer(case: Case, response: dict) -> dict:
    text = response["choices"][0]["message"].get("content") or ""
    codes = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])", text.lower()))
    usage = response["usage"]
    if int(usage["prompt_tokens"]) != case.prompt_tokens:
        raise ValueError("Tokenize and inference prompt lengths differ")
    cached = int(usage["prompt_tokens_details"]["cached_tokens"])
    return {
        "case_id": case.case_id,
        "prompt_sha256": case.digest(),
        "expected": case.answer,
        "response": text,
        "correct": codes == {case.answer},
        "cached_tokens": cached,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "needle_end_tokens": case.needle_end_tokens,
        "finish_reason": response["choices"][0]["finish_reason"],
    }


def verify_restore(before: dict, after: dict, rows: list[dict]) -> dict:
    """Accept a wave only when each response and counters prove external reuse."""
    delta = {name: after[name] - before[name] for name in METRICS[:3]}
    cached = sum(row["cached_tokens"] for row in rows)
    if any(row["cached_tokens"] <= 0 for row in rows):
        raise ValueError("A measured request did not report cached tokens")
    if any(not 0 < row["needle_end_tokens"] <= row["cached_tokens"] for row in rows):
        raise ValueError("A needle lies outside the restored CPU prefix")
    if any(row["prompt_tokens"] <= 0 for row in rows):
        raise ValueError("A measured request has invalid token accounting")
    if delta["prefix_cache_hits_total"] != 0:
        raise ValueError("Measured wave reused native NPU cache")
    if delta["external_prefix_cache_hits_total"] != cached:
        raise ValueError("External hit delta does not match response cached tokens")
    if delta["num_preemptions_total"] != 0:
        raise ValueError("Measured wave was preempted")
    return delta


def run(options) -> dict:
    if not options.dedicated_test_server:
        raise ValueError(
            "Use a dedicated test server; this run resets its NPU prefix cache"
        )
    if options.output.exists():
        raise FileExistsError(options.output)
    client = Client(options.base_url, options.model, options.timeout)
    check_control_api(client.request("/openapi.json"))
    report = {
        "mode": options.mode,
        "model": options.model,
        "suite_id": options.suite_id,
        "context_tokens": options.context_tokens,
        "config_sha256": hashlib.sha256(options.config.read_bytes()).hexdigest(),
        "waves": [],
        "complete": False,
    }
    try:
        for concurrency in options.concurrencies:
            for depth in options.depths:
                for repeat in range(options.repeats):
                    cases = [
                        make_case(
                            client,
                            f"{options.suite_id}-c{concurrency}-d{depth}-r{repeat}-i{i}",
                            options.context_tokens,
                            depth,
                        )
                        for i in range(concurrency)
                    ]
                    for case in cases:
                        client.complete(
                            case.prefix, "Confirm that the document was received.", 8
                        )
                    time.sleep(options.store_settle_seconds)
                    clear_npu_only(client, options.timeout)
                    before = wait_idle(client, options.timeout)
                    barrier = threading.Barrier(concurrency)

                    def measured(case, barrier=barrier):
                        barrier.wait(timeout=options.timeout)
                        return check_answer(
                            case, client.complete(case.prefix, case.question)
                        )

                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(measured, cases))
                    after = wait_idle(client, options.timeout)
                    wave = {
                        "concurrency": concurrency,
                        "depth": depth,
                        "repeat": repeat,
                        "before": before,
                        "after": after,
                        "requests": rows,
                        "restore_verified": False,
                    }
                    report["waves"].append(wave)
                    wave["delta"] = verify_restore(before, after, rows)
                    wave["restore_verified"] = True
        report["complete"] = True
        return report
    finally:
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def compare(sync: dict, asynchronous: dict) -> dict:
    """Require matched inputs and independently verified CPU-restore waves."""
    if sync.get("mode") != "sync" or asynchronous.get("mode") != "async":
        raise ValueError("Expected a sync report followed by an async report")
    for report in (sync, asynchronous):
        if not report.get("complete") or not report["waves"]:
            raise ValueError("Incomplete correctness run")
        if not any(wave["concurrency"] > 1 for wave in report["waves"]):
            raise ValueError("Correctness evaluation must include concurrent requests")
        for wave in report["waves"]:
            cached = sum(row["cached_tokens"] for row in wave["requests"])
            if (
                not wave.get("restore_verified")
                or any(row["cached_tokens"] <= 0 for row in wave["requests"])
                or any(
                    not 0 < row["needle_end_tokens"] <= row["cached_tokens"]
                    for row in wave["requests"]
                )
                or wave["delta"]["external_prefix_cache_hits_total"] != cached
                or wave["delta"]["prefix_cache_hits_total"] != 0
                or wave["delta"]["num_preemptions_total"] != 0
            ):
                raise ValueError("Report contains an unverified CPU-restore wave")
    left = [row for wave in sync["waves"] for row in wave["requests"]]
    right = [row for wave in asynchronous["waves"] for row in wave["requests"]]
    if sync["model"] != asynchronous["model"] or len(left) != len(right):
        raise ValueError("Model or case count differs")
    for a, b in zip(left, right, strict=True):
        if any(a[key] != b[key] for key in ("case_id", "prompt_sha256", "expected")):
            raise ValueError("Sync/async inputs differ")
    return {
        "cases": len(left),
        "sync_correct": sum(row["correct"] for row in left),
        "async_correct": sum(row["correct"] for row in right),
        "all_correct": all(row["correct"] for row in left + right),
        "cpu_restore_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--mode", choices=("sync", "async"), required=True)
    execute.add_argument("--base-url", default="http://127.0.0.1:8010")
    execute.add_argument("--model", required=True)
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--suite-id", required=True)
    execute.add_argument("--context-tokens", type=int, default=32768)
    execute.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    execute.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    execute.add_argument("--repeats", type=int, default=1)
    execute.add_argument("--timeout", type=float, default=300)
    execute.add_argument("--store-settle-seconds", type=float, default=2)
    execute.add_argument("--dedicated-test-server", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("sync", type=Path)
    comparison.add_argument("asynchronous", type=Path)
    options = parser.parse_args()
    if options.command == "run":
        if (
            options.context_tokens < 1024
            or options.repeats < 1
            or options.timeout <= 0
            or options.store_settle_seconds < 0
            or any(value < 1 for value in options.concurrencies)
            or any(not 0 < value < 1 for value in options.depths)
        ):
            parser.error("Invalid context length, repeats, concurrency, or depth")
        result = run(options)
        passed = all(
            row["correct"] for wave in result["waves"] for row in wave["requests"]
        )
        print(json.dumps({"output": str(options.output), "all_correct": passed}))
    else:
        result = compare(
            json.loads(options.sync.read_text(encoding="utf-8")),
            json.loads(options.asynchronous.read_text(encoding="utf-8")),
        )
        passed = result["all_correct"]
        print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
