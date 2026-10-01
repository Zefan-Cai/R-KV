"""Scoring regressions that run without loading Torch or model weights."""

import ast
import re
import sys
from decimal import Decimal, InvalidOperation
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
namespace = {"Decimal": Decimal, "InvalidOperation": InvalidOperation,
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


def test_primary_scoring_rejects_unfinished_reasoning_and_incomplete_answers():
    # Isolate the protocol from optional symbolic-grader dependencies. The fake
    # parser deliberately accepts unclosed boxes just like the repository parser.
    parser = ModuleType("parser")
    def extract_answer(text, dataset, use_last_number=True):
        assert not use_last_number
        if "boxed{" in text:
            return text.rsplit("boxed{", 1)[1].split("}", 1)[0]
        if "final answer is " in text:
            return text.rsplit("final answer is ", 1)[1].strip()
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
                "thinking": thinking, "answer": "Work #### 1000"}
               for i, (output, model_type, thinking) in enumerate(outputs)]
    with patch.dict(sys.modules, {"parser": parser, "evaluate": evaluate}):
        scored, result = score(records, "gsm8k")
    assert [record["score"][0] for record in scored] == [False, True, True, False, False, False, False]
    assert result["unfinished_reasoning"] == 1
    assert result["empty_samples"] == 3
