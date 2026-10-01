"""Scoring regressions that run without loading Torch or model weights."""

import ast
import re
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


source = Path(__file__).resolve().parents[1] / "run_hybrid_math.py"
tree = ast.parse(source.read_text())
helpers = ast.Module(
    body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
          and node.name in {"final_response", "gsm8k_exact_match", "score"}],
    type_ignores=[],
)
namespace = {"Decimal": Decimal, "InvalidOperation": InvalidOperation, "Fraction": Fraction,
             "re": re, "sys": sys, "ROOT": source.parent}
exec(compile(helpers, str(source), "exec"), namespace)
final_response = namespace["final_response"]
gsm8k_exact_match = namespace["gsm8k_exact_match"]
score = namespace["score"]


def test_qwen_prompt_opens_reasoning_and_truncated_thought_is_not_an_answer():
    text = "We calculate 20. The intermediate answer is \\boxed{20}."
    assert final_response(text, "qwen3_5_text", True) == ("", False)
    assert final_response(text + "</think>Final answer: \\boxed{21}<|im_end|>",
                          "qwen3_5_text", True) == ("Final answer: \\boxed{21}", True)


def test_qwen_non_thinking_can_return_a_direct_answer():
    assert final_response("Final answer: \\boxed{20}<|im_end|>",
                          "qwen3_5_text", False) == ("Final answer: \\boxed{20}", True)


def test_gemma4_thought_answers_are_excluded_and_final_channel_survives():
    output = "<|channel>thought\nTentative answer \\boxed{20}<channel|>Final answer: \\boxed{21}<turn|>"
    assert final_response(output, "gemma4_text", True) == ("Final answer: \\boxed{21}", True)
    assert final_response("<|channel>thought\nTentative answer \\boxed{20}<turn|>",
                          "gemma4_text", True) == ("", False)
    assert final_response("<|channel>thought\n<channel|><|channel>final\n\\boxed{21}<channel|><turn|>",
                          "gemma4_text", False) == ("\\boxed{21}", True)


def test_gemma4_direct_answers_do_not_require_a_thought_block():
    assert final_response("\\boxed{20}<turn|>", "gemma4_text", True) == ("\\boxed{20}", True)


def test_gsm8k_numeric_equality_rejects_percent_scaling_and_tolerance():
    assert gsm8k_exact_match("1,000.00", "1000")
    assert gsm8k_exact_match("-0.5", "-.50")
    assert not gsm8k_exact_match("10", "1000")
    assert not gsm8k_exact_match("100000", "1000")
    assert not gsm8k_exact_match("1000.01", "1000")
    assert not gsm8k_exact_match("", "0")
    assert not gsm8k_exact_match("NaN", "NaN")


def test_gsm8k_exact_rationals_accept_equivalent_numeric_forms():
    assert gsm8k_exact_match(r"\frac{1}{2}", "0.5")
    assert gsm8k_exact_match(r"-\dfrac{1}{2}", "-0.5")
    assert gsm8k_exact_match(r"\tfrac{-1}{2}", "-0.5")
    assert gsm8k_exact_match("12 / 3", "4.0")
    assert not gsm8k_exact_match("1 / 3", "0.3333")
    assert not gsm8k_exact_match(r"\frac{1}{0}", "0")
    assert not gsm8k_exact_match("Infinity", "Infinity")


def test_gsm8k_whole_hour_clock_form_is_exact_without_rounding():
    assert gsm8k_exact_match("2:00 PM", "2")
    assert gsm8k_exact_match("02:00pm", "2")
    assert gsm8k_exact_match("2 PM", "2")
    assert gsm8k_exact_match("12:00 AM", "12")
    assert gsm8k_exact_match("2:00", "2")
    assert gsm8k_exact_match("14:00", "14")
    assert gsm8k_exact_match("00:00", "0")
    assert not gsm8k_exact_match("2:30 PM", "2")
    assert not gsm8k_exact_match("14:00 PM", "2")
    assert not gsm8k_exact_match("2:00 PM", "14")
    assert not gsm8k_exact_match("24:00", "24")
    assert not gsm8k_exact_match("00:00 AM", "0")


def test_primary_scoring_rejects_unfinished_reasoning_and_incomplete_answers():
    # Isolate the protocol from optional symbolic-grader dependencies. The fake
    # parser deliberately accepts unclosed boxes just like the repository parser.
    parser = ModuleType("parser")
    def extract_answer(text, dataset, use_last_number=True):
        if "boxed{" in text:
            return text.rsplit("boxed{", 1)[1].split("}", 1)[0]
        if "final answer is " in text:
            return text.rsplit("final answer is ", 1)[1].strip()
        if use_last_number:
            numbers = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
            return numbers[-1] if numbers else ""
        return ""
    parser.extract_answer = extract_answer
    parser.parse_ground_truth = lambda record, dataset: record["answer"].split("####")
    evaluate = ModuleType("evaluate")
    evaluate.evaluate = lambda **kwargs: (_ for _ in ()).throw(AssertionError("GSM must use exact numeric scoring"))
    outputs = [
        ("\\boxed{1000}", "qwen3_5_text", True),
        ("<|channel>thought\n\\boxed{20}<channel|>Final answer: \\boxed{1000}", "gemma4_text", True),
        ("Final answer: 1000", "gemma4_text", False),
        ("Final answer: \\boxed{1000", "gemma4_text", False),
        ("An intermediate value is 1000", "gemma4_text", False),
        ("\\boxed{1000.01}", "gemma4_text", False),
        ("\\boxed{10}", "gemma4_text", False),
    ]
    records = [{"idx": i, "output": output, "model_type": model_type,
                "thinking": thinking, "truncated": i == 4, "answer": "Work #### 1000"}
               for i, (output, model_type, thinking) in enumerate(outputs)]
    with patch.dict(sys.modules, {"parser": parser, "evaluate": evaluate}):
        scored, result = score(records, "gsm8k")
    assert [record["score"][0] for record in scored] == [False, True, True, False, False, False, False]
    assert result["unfinished_reasoning"] == 1
    assert result["empty_samples"] == 3


def test_finished_unboxed_gsm_answers_score_separately_from_format_compliance():
    parser = ModuleType("parser")
    def extract_answer(text, dataset, use_last_number=True):
        if "boxed{" in text:
            return text.rsplit("boxed{", 1)[1].split("}", 1)[0]
        numbers = re.findall(r"-?\d*\.?\d+", text.replace(",", "")) if use_last_number else []
        return numbers[-1] if numbers else ""
    parser.extract_answer = extract_answer
    parser.parse_ground_truth = lambda record, dataset: record["answer"].split("####")
    evaluate = ModuleType("evaluate")
    evaluate.evaluate = lambda **kwargs: (_ for _ in ()).throw(AssertionError("GSM must use exact numeric scoring"))
    records = [
        {"output": "James has 16 Facebook friends.", "answer": "Work #### 16"},
        {"output": "Total calories = 40,000 + 45,000 = 85,000 calories.", "answer": "Work #### 85000"},
        {"output": "Intermediate value is 16", "answer": "Work #### 16", "truncated": True},
        {"output": "\\boxed{16", "answer": "Work #### 16"},
        {"output": "<|channel>thought\n16", "answer": "Work #### 16", "thinking": True},
        {"output": "16", "answer": "Work #### 16", "thinking": True, "model_type": "qwen3_5_text"},
    ]
    records = [{"idx": i, "model_type": "gemma4_text", "thinking": False,
                "truncated": False, **record} for i, record in enumerate(records)]
    with patch.dict(sys.modules, {"parser": parser, "evaluate": evaluate}):
        scored, result = score(records, "gsm8k")
    assert [r["score"][0] for r in scored] == [True, True, False, False, False, False]
    assert result["fallback_answers"] == 2
    assert result["answer_format_failures"] == 6
    assert scored[0]["format_pred"] == ""
    assert scored[0]["pred"] == ["16"]


def test_finished_unboxed_math_uses_repository_fallback_but_capped_math_does_not():
    parser = ModuleType("parser")
    parser.extract_answer = lambda text, dataset, use_last_number=True: (
        re.findall(r"\d+", text)[-1] if use_last_number and re.findall(r"\d+", text) else "")
    parser.parse_ground_truth = lambda record, dataset: ("", "7")
    evaluate = ModuleType("evaluate")
    def evaluate_math(*, data_name, prompt_type, samples):
        assert data_name == "math"
        for record in samples:
            record["score"] = [record["pred"][0] == "7"]
        return samples, {"num_samples": len(samples), "acc": 50.0}
    evaluate.evaluate = evaluate_math
    records = [{"idx": i, "output": "The result is 7.", "model_type": "gemma4_text",
                "thinking": False, "truncated": bool(i)} for i in range(2)]
    with patch.dict(sys.modules, {"parser": parser, "evaluate": evaluate}):
        scored, result = score(records, "math")
    assert [r["score"][0] for r in scored] == [True, False]
    assert result["fallback_answers"] == 1
