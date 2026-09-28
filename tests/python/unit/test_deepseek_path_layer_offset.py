"""Both runtime layer-offset guards must accept local mixed-case model paths."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "moe_infinity/runtime/model_offload.py"
)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("deepseek-ai/DeepSeek-V2-Lite", True),
        ("/home/user/models/DeepSeek-V2-Lite", True),
        ("/models/DEEPSEEK-V2-LITE", True),
        ("/models/GLM-5", True),
        ("/models/Mixtral-8x7B", False),
    ],
)
def test_routed_layer_offset_recognizes_case_insensitive_paths(
    model_name, expected
):
    tree = ast.parse(SOURCE.read_text())
    guards = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "'deepseek'" in ast.unparse(node.test)
        and "model_name" in ast.unparse(node.test)
        and "'glm'" in ast.unparse(node.test)
    ]
    assert len(guards) == 2
    for guard in guards:
        code = compile(ast.Expression(guard), str(SOURCE), "eval")
        actual = eval(
            code,
            {"__builtins__": {}},
            {
                "model_name": model_name,
                "self": SimpleNamespace(model_name=model_name),
            },
        )
        assert actual is expected
