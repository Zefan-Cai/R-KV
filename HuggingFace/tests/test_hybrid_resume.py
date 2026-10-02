"""Protect full test coverage and interrupted generation checkpoints."""

import ast
import hashlib
import json
import random
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "run_hybrid_math.py"
tree = ast.parse(source.read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"load_dataset", "resume_outputs"}], type_ignores=[])
namespace = {"json": json, "hashlib": hashlib, "random": random}
exec(compile(helpers, str(source), "exec"), namespace)
load_dataset = namespace["load_dataset"]
resume_outputs = namespace["resume_outputs"]
protocol = {"model": "checkpoint", "mode": "rkv128", "thinking": False}


def generated(record):
    return {**record, **protocol, "output": "done", "prefill_tokens": 8, "output_tokens": 4,
            "generation_seconds": 1.0, "tokens_per_second": 4.0, "peak_allocated_gib": 1.0,
            "baseline_allocated_gib": 0.5, "incremental_peak_gib": 0.5, "truncated": False,
            "adapter_stats": {"compression_count": 1}}


def dataset_file(tmp_path, count, dataset="gsm8k"):
    path = tmp_path / (dataset + ".jsonl")
    fields = ("question", "answer") if dataset == "gsm8k" else ("problem", "solution")
    path.write_text("".join(json.dumps({fields[0]: str(i), fields[1]: "reference"}) + "\n"
                            for i in range(count)))
    return path


def test_full_math_rejects_math500_and_full_request_rejects_sampling(tmp_path):
    path = dataset_file(tmp_path, 500, "math")
    with pytest.raises(ValueError, match="Full math"):
        load_dataset(path, "math", 0, 42, 0, 1, True)
    path = dataset_file(tmp_path, 1319)
    with pytest.raises(ValueError, match="sample-limit=0"):
        load_dataset(path, "gsm8k", 100, 42, 0, 1, True)


@pytest.mark.parametrize("dataset,count", [("gsm8k", 1319), ("math", 5000)])
def test_full_shards_cover_every_id_once_with_same_data_hash(tmp_path, dataset, count):
    path = dataset_file(tmp_path, count, dataset)
    shards = [load_dataset(path, dataset, 0, 42, i, 3, True) for i in range(3)]
    ids = [r["idx"] for records, _ in shards for r in records]
    assert sorted(ids) == list(range(count))
    assert len(set(manifest["data_sha256"] for _, manifest in shards)) == 1
    assert all(manifest["dataset_count"] == count and manifest["full_benchmark"]
               and len(records) == manifest["expected_shard_count"] for records, manifest in shards)


def test_resume_repairs_only_incomplete_last_record_and_preserves_it(tmp_path):
    records, _ = load_dataset(dataset_file(tmp_path, 3), "gsm8k", 0, 42, 0, 1, False)
    path = tmp_path / "outputs.jsonl"
    result = generated(records[0])
    valid = json.dumps(result) + "\n"
    partial = b'{"idx": 1, "output": "unfinished'
    path.write_bytes(valid.encode() + partial)
    assert resume_outputs(path, records, protocol) == [result]
    assert path.read_text() == valid
    assert path.with_suffix(".jsonl.partial").read_bytes() == partial


def test_resume_inserts_missing_newline_after_valid_last_record(tmp_path):
    records, _ = load_dataset(dataset_file(tmp_path, 2), "gsm8k", 0, 42, 0, 1, False)
    path = tmp_path / "outputs.jsonl"
    path.write_text(json.dumps(generated(records[0])))
    assert resume_outputs(path, records, protocol) == [generated(records[0])]
    assert path.read_bytes().endswith(b"\n")
    with path.open("a") as stream:
        stream.write(json.dumps(generated(records[1])) + "\n")
    assert len(resume_outputs(path, records, protocol)) == 2


@pytest.mark.parametrize("failure", ["duplicate", "foreign", "changed", "corrupt_completed", "corrupt_middle"])
def test_resume_rejects_ambiguous_or_different_completed_data(tmp_path, failure):
    records, _ = load_dataset(dataset_file(tmp_path, 3), "gsm8k", 0, 42, 0, 1, False)
    first = json.dumps(generated(records[0])) + "\n"
    tails = {"duplicate": first, "foreign": json.dumps({**generated(records[1]), "idx": 99}) + "\n",
             "changed": json.dumps({**generated(records[1]), "example_sha256": "old"}) + "\n",
             "corrupt_completed": "{bad}\n", "corrupt_middle": "{bad}\n" + json.dumps(records[1])}
    path = tmp_path / "outputs.jsonl"
    path.write_text(first + tails[failure])
    before = path.read_bytes()
    with pytest.raises(ValueError):
        resume_outputs(path, records, protocol)
    assert path.read_bytes() == before


@pytest.mark.parametrize("key,value", [("model", "other"), ("mode", "fullkv"), ("thinking", True)])
def test_resume_rejects_records_from_another_model_or_arm(tmp_path, key, value):
    records, _ = load_dataset(dataset_file(tmp_path, 1), "gsm8k", 0, 42, 0, 1, False)
    path = tmp_path / "outputs.jsonl"
    path.write_text(json.dumps({**generated(records[0]), key: value}) + "\n")
    with pytest.raises(ValueError, match="Wrong generation protocol"):
        resume_outputs(path, records, protocol)


def test_full_input_rejects_repeated_source_records(tmp_path):
    path = dataset_file(tmp_path, 5000, "math")
    path.write_text((json.dumps({"problem": "same", "solution": "same"}) + "\n") * 5000)
    with pytest.raises(ValueError, match="Duplicate source"):
        load_dataset(path, "math", 0, 42, 0, 1, True)
