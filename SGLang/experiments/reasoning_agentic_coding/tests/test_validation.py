#!/usr/bin/env python3
"""Tests for strict campaign completion-marker validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import summarize_server_log  # noqa: E402
import validate_bfcl_pilot  # noqa: E402
import validate_evalplus  # noqa: E402
import evalplus_codegen  # noqa: E402


class ServerSummaryTest(unittest.TestCase):
    def test_accepts_matching_decode_compaction(self) -> None:
        summary = summarize_server_log.summarize(
            "The server is fired up and ready to roll!\n"
            "R-KV compacted req_pool_idx=0: phys 5000 -> 4096 slots (freed 904)\n"
        )
        summarize_server_log.validate(summary, "d-4k")

    def test_rejects_missing_prefill_compaction(self) -> None:
        summary = summarize_server_log.summarize(
            "The server is fired up and ready to roll!\n"
        )
        with self.assertRaisesRegex(ValueError, "no compaction"):
            summarize_server_log.validate(summary, "p-4k")


class BfclValidationTest(unittest.TestCase):
    def make_root(self, result: str = "ok") -> Path:
        root = Path(self.temp_dir.name)
        (root / "result/model/agentic/memory/kv").mkdir(parents=True)
        (root / "score/model/agentic/memory/kv").mkdir(parents=True)
        manifest = {"memory_kv": ["memory_kv_prereq_0", "memory_kv_1"]}
        (root / "test_case_ids_to_generate.json").write_text(json.dumps(manifest))
        result_path = (
            root / "result/model/agentic/memory/kv/BFCL_v4_memory_kv_result.json"
        )
        entries = [
            {"id": "memory_kv_prereq_0", "result": "setup"},
            {"id": "memory_kv_1", "result": result},
        ]
        result_path.write_text("".join(json.dumps(value) + "\n" for value in entries))
        score_path = root / "score/model/agentic/memory/kv/BFCL_v4_memory_kv_score.json"
        score_path.write_text(
            json.dumps({"accuracy": 1.0, "correct_count": 1, "total_count": 1})
            + "\n"
        )
        return root

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_accepts_complete_artifacts(self) -> None:
        summary = validate_bfcl_pilot.validate(self.make_root())
        self.assertTrue(summary["valid"])

    def test_rejects_swallowed_inference_error(self) -> None:
        root = self.make_root("Error during inference: timeout")
        with self.assertRaisesRegex(ValueError, "inference errors"):
            validate_bfcl_pilot.validate(root)


class EvalPlusValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_artifacts(
        self,
        sample_ids: list[str],
        evaluated_ids: list[str],
        *,
        status: str = "pass",
    ) -> None:
        samples = self.root / "humaneval/model_openai_temp_0.0.jsonl"
        samples.parent.mkdir(parents=True)
        samples.write_text(
            "".join(
                json.dumps({"task_id": task_id, "solution": "pass"}) + "\n"
                for task_id in sample_ids
            )
        )
        results = samples.with_name("model_openai_temp_0.0_eval_results.json")
        results.write_text(
            json.dumps(
                {
                    "eval": {
                        task_id: [
                            {
                                "base_status": status,
                                "plus_status": status,
                            }
                        ]
                        for task_id in evaluated_ids
                    }
                }
            )
        )

    def test_accepts_matching_complete_coverage(self) -> None:
        self.write_artifacts(["HumanEval/0", "HumanEval/1"], ["HumanEval/0", "HumanEval/1"])
        summary = validate_evalplus.validate(self.root, expected_tasks=2)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["base_passes"], 2)
        self.assertEqual(summary["plus_passes"], 2)

    def test_rejects_duplicate_samples(self) -> None:
        self.write_artifacts(["HumanEval/0", "HumanEval/0"], ["HumanEval/0"])
        with self.assertRaisesRegex(ValueError, "duplicate samples"):
            validate_evalplus.validate(self.root, expected_tasks=2)

    def test_rejects_score_coverage_mismatch(self) -> None:
        self.write_artifacts(["HumanEval/0", "HumanEval/1"], ["HumanEval/0"])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_evalplus.validate(self.root, expected_tasks=2)

    def test_rejects_all_timeout_infrastructure_failure(self) -> None:
        self.write_artifacts(
            ["HumanEval/0", "HumanEval/1"],
            ["HumanEval/0", "HumanEval/1"],
            status="timeout",
        )
        with self.assertRaisesRegex(ValueError, "infrastructure failure"):
            validate_evalplus.validate(self.root, expected_tasks=2)


class EvalPlusCodegenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "samples.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_repairs_only_malformed_trailing_record(self) -> None:
        self.path.write_text(
            json.dumps({"task_id": "HumanEval/0", "solution": "pass"})
            + "\n{\"task_id\":",
            encoding="utf-8",
        )
        existing = evalplus_codegen.load_existing(self.path)
        self.assertEqual(set(existing), {"HumanEval/0"})
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_normalizes_valid_record_without_newline(self) -> None:
        self.path.write_text(
            json.dumps({"task_id": "HumanEval/0", "solution": "pass"}),
            encoding="utf-8",
        )
        evalplus_codegen.load_existing(self.path)
        self.assertTrue(self.path.read_text(encoding="utf-8").endswith("\n"))

    def test_rejects_duplicate_task_ids(self) -> None:
        row = json.dumps({"task_id": "HumanEval/0", "solution": "pass"})
        self.path.write_text(f"{row}\n{row}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            evalplus_codegen.load_existing(self.path)

    def test_generate_resumes_only_missing_tasks(self) -> None:
        self.path.write_text(
            json.dumps({"task_id": "HumanEval/0", "solution": "existing"}) + "\n",
            encoding="utf-8",
        )
        dataset = {
            "HumanEval/0": {"prompt": "def zero():", "entry_point": "zero"},
            "HumanEval/1": {"prompt": "def one():", "entry_point": "one"},
        }
        response = {"choices": [{"message": {"content": "```python\nreturn 1\n```"}}]}
        with mock.patch.object(
            evalplus_codegen, "request_completion", return_value=response
        ) as request:
            summary = evalplus_codegen.generate(
                base_url="http://server/v1",
                model="model",
                output=self.path,
                concurrency=2,
                max_tokens=768,
                dataset=dataset,
                sanitizer=lambda value, **_: f"sanitized:{value}",
            )
        self.assertEqual(
            summary,
            {"existing": 1, "generated": 1, "legacy_missing_raw": 1, "total": 2},
        )
        self.assertEqual(request.call_count, 1)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual({row["task_id"] for row in rows}, set(dataset))

    def test_openai_payload_matches_evalplus_provider(self) -> None:
        messages = [
            {"role": "system", "content": evalplus_codegen.SYSTEM_MESSAGE},
            {"role": "user", "content": "prompt"},
        ]
        with mock.patch.object(evalplus_codegen, "request_json", return_value={}) as request:
            evalplus_codegen.request_completion(
                "http://server/v1", "model", messages, max_tokens=768
            )
        request.assert_called_once_with(
            "http://server/v1/chat/completions",
            {
                "model": "model",
                "messages": messages,
                "max_tokens": 768,
                "temperature": 0.0,
                "n": 1,
                "top_p": 0.95,
            },
        )

    def test_drops_orphan_raw_record_before_resume(self) -> None:
        raw_path = self.path.with_name("samples.raw.jsonl")
        raw_path.write_text(
            json.dumps({"task_id": "HumanEval/1", "solution": "orphan"}) + "\n",
            encoding="utf-8",
        )
        dataset = {"HumanEval/1": {"prompt": "def one():", "entry_point": "one"}}
        response = {"choices": [{"message": {"content": "solution"}}]}
        with mock.patch.object(
            evalplus_codegen, "request_completion", return_value=response
        ):
            evalplus_codegen.generate(
                base_url="http://server/v1",
                model="model",
                output=self.path,
                concurrency=1,
                max_tokens=768,
                dataset=dataset,
                sanitizer=lambda value, **_: value,
            )
        raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        self.assertEqual(raw_rows, [{"task_id": "HumanEval/1", "solution": "solution"}])


if __name__ == "__main__":
    unittest.main()
