#!/usr/bin/env python3
"""Tests for strict campaign completion-marker validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import summarize_server_log  # noqa: E402
import validate_bfcl_pilot  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
