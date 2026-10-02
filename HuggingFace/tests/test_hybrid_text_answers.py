"""Full MATH has two text references that must survive unit stripping."""

import ast
import re
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "evaluation/parser.py"
tree = ast.parse(source.read_text())
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"strip_string", "extract_answer", "_fix_fracs",
                                           "_fix_a_slash_b", "_fix_sqrt"}], type_ignores=[])
namespace = {"re": re, "unit_texts": ["east", "west", "cm", "hours"],
             "convert_word_number": lambda text: text}
exec(compile(helpers, str(source), "exec"), namespace)
extract_answer = namespace["extract_answer"]


@pytest.mark.parametrize("output,expected", [
    (r"\boxed{\mbox{Saturday}}", "Saturday"),
    (r"\boxed{\text{east}}", "east"),
    (r"\boxed{east}", "east"),
    (r"\boxed{\text{west}}", "west"),
    (r"\boxed{3\text{ cm}}", "3"),
])
def test_text_labels_survive_and_numeric_unit_suffixes_still_strip(output, expected):
    assert extract_answer(output, "math", use_last_number=False) == expected
