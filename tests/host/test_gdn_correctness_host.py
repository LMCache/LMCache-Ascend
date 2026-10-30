# SPDX-License-Identifier: Apache-2.0
"""Host tests for the correctness client; no model or device is exercised."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import json
import threading
import unittest

# Third Party
from _gdn_source import ROOT, source_module


def control_spec():
    def parameter(name, **schema):
        return {"name": name, "in": "query", "schema": schema}

    return {
        "paths": {
            "/pause": {
                "post": {
                    "parameters": [
                        parameter("mode", enum=["wait", "abort"]),
                        parameter("clear_cache", type="boolean"),
                    ]
                }
            },
            "/resume": {"post": {}},
            "/tokenize": {"post": {}},
            "/reset_prefix_cache": {
                "post": {
                    "parameters": [
                        parameter("reset_running_requests", type="boolean"),
                        parameter("reset_external", type="boolean"),
                    ]
                }
            },
        }
    }


class CorrectnessClientTest(unittest.TestCase):
    def setUp(self):
        self.context = source_module(
            "examples/gdn/validate_async_correctness.py", "gdn_correctness_test"
        )
        self.module = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def counters(self, external=0, native=0):
        return {
            "prefix_cache_hits_total": native,
            "external_prefix_cache_hits_total": external,
            "num_preemptions_total": 0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
        }

    def test_metrics_require_proof_counters_and_reject_invalid_values(self):
        raw = "\n".join(f"vllm:{name} 0" for name in self.module.METRICS)
        self.assertEqual(self.module.metrics(raw), self.counters())
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.module.metrics(
                raw.replace("external_prefix_cache_hits_total", "absent")
            )
        with self.assertRaisesRegex(ValueError, "Invalid"):
            self.module.metrics(raw.replace("hits_total 0", "hits_total 1e999"))

    def test_restore_rejects_native_hits_missing_external_hits_and_preemption(self):
        rows = [{"cached_tokens": 256, "prompt_tokens": 1024, "needle_end_tokens": 128}]
        before = self.counters()
        self.module.verify_restore(before, self.counters(external=256), rows)
        for after in (
            self.counters(external=256, native=1),
            self.counters(external=0),
            {**self.counters(external=256), "num_preemptions_total": 1},
        ):
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.module.verify_restore(before, after, rows)
        with self.assertRaises(ValueError):
            self.module.verify_restore(
                before,
                self.counters(external=256),
                rows
                + [
                    {
                        "cached_tokens": 0,
                        "prompt_tokens": 1024,
                        "needle_end_tokens": 128,
                    }
                ],
            )
        rows[0]["needle_end_tokens"] = 512
        with self.assertRaisesRegex(ValueError, "outside"):
            self.module.verify_restore(before, self.counters(external=256), rows)

    def test_control_api_requires_explicit_non_destructive_flags(self):
        self.module.check_control_api(control_spec())
        spec = control_spec()
        spec["paths"]["/reset_prefix_cache"]["post"]["parameters"].pop()
        with self.assertRaises(ValueError):
            self.module.check_control_api(spec)

    def test_refused_npu_reset_still_resumes_and_never_resets_external(self):
        client = Mock()
        client.request.side_effect = [None, False, None]
        with patch.object(self.module, "wait_idle"):
            with self.assertRaisesRegex(RuntimeError, "refused"):
                self.module.clear_npu_only(client, 1)
        paths = [call.args[0] for call in client.request.call_args_list]
        self.assertEqual(
            paths,
            [
                "/pause?mode=wait&clear_cache=false",
                "/reset_prefix_cache?reset_running_requests=false&reset_external=false",
                "/resume",
            ],
        )

    def test_cases_are_deterministic_and_question_does_not_include_answer(self):
        client = NS(
            token_count=lambda prefix, question: len((prefix + question).split()) + 32
        )
        first = self.module.make_case(client, "suite-c2-d0.5-r0-i0", 1024, 0.5)
        second = self.module.make_case(client, "suite-c2-d0.5-r0-i0", 1024, 0.5)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.prompt_tokens, 1024)
        self.assertIn(first.answer, first.prefix)
        self.assertNotIn(first.answer, first.question)

    def test_answer_scoring_requires_the_expected_passkey(self):
        case = self.module.Case(
            "r", "context", "question", "0123456789abcdef", 1024, 128
        )
        response = {
            "choices": [{"message": {"content": case.answer}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1024,
                "prompt_tokens_details": {"cached_tokens": 256},
            },
        }
        self.assertTrue(self.module.check_answer(case, response)["correct"])
        response["choices"][0]["message"]["content"] = "fedcba9876543210"
        self.assertFalse(self.module.check_answer(case, response)["correct"])

    def run_fake_server(self):
        module = self.module

        class Server:
            def __init__(self):
                self.hits = 0
                self.lock = threading.Lock()

            def request(self, path, body=None, *, post=False):
                if path == "/openapi.json":
                    return control_spec()
                if path == "/metrics":
                    values = {name: 0 for name in module.METRICS}
                    values["external_prefix_cache_hits_total"] = self.hits
                    return "\n".join(
                        f"vllm:{key} {value}" for key, value in values.items()
                    )
                return True

            def token_count(self, prefix, question):
                return len((prefix + question).split()) + 32

            def complete(self, prefix, question, max_tokens=128):
                if question.startswith("Confirm"):
                    return {}
                answer = module.re.search(r"secret code ([0-9a-f]{16})", prefix).group(
                    1
                )
                with self.lock:
                    self.hits += 768
                return {
                    "choices": [
                        {"message": {"content": answer}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": self.token_count(prefix, question),
                        "prompt_tokens_details": {"cached_tokens": 768},
                    },
                }

        server = Server()
        output = Mock(spec=Path)
        output.exists.return_value = False
        options = NS(
            dedicated_test_server=True,
            output=output,
            base_url="http://127.0.0.1:8010",
            model="test",
            timeout=2,
            mode="sync",
            suite_id="host",
            context_tokens=1024,
            config=ROOT / "README.md",
            concurrencies=[1, 2],
            depths=[0.5],
            repeats=1,
            store_settle_seconds=0,
        )
        with (
            patch.object(module, "Client", return_value=server),
            patch.object(module.time, "sleep"),
        ):
            report = module.run(options)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["waves"]), 2)
        saved = json.loads(output.write_text.call_args.args[0])
        self.assertEqual(saved, report)
        return report

    def test_full_client_warms_resets_and_measures_real_concurrent_host_threads(self):
        report = self.run_fake_server()
        rows = [row for wave in report["waves"] for row in wave["requests"]]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["correct"] for row in rows))
        self.assertEqual(
            report["waves"][1]["delta"]["external_prefix_cache_hits_total"], 1536
        )

    def test_comparison_rejects_changed_inputs_incomplete_runs_and_wrong_answers(self):
        report = self.run_fake_server()
        other = json.loads(json.dumps(report))
        other["mode"] = "async"
        self.assertTrue(self.module.compare(report, other)["all_correct"])
        other["waves"][0]["requests"][0]["correct"] = False
        self.assertFalse(self.module.compare(report, other)["all_correct"])
        other["waves"][0]["requests"][0]["prompt_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.module.compare(report, other)
        other["complete"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.module.compare(report, other)


if __name__ == "__main__":
    unittest.main()
