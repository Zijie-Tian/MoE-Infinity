import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SETUP_PATH = ROOT / "setup.py"


def _load_find_cuda_home(torch_cuda_version):
    tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_find_cuda_home"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "os": __import__("os"),
        "torch": SimpleNamespace(
            version=SimpleNamespace(cuda=torch_cuda_version)
        ),
        "torch_available": True,
    }
    exec(compile(module, str(SETUP_PATH), "exec"), namespace)
    return namespace["_find_cuda_home"]


@pytest.mark.parametrize("torch_cuda_version", ["12.8", "13.0"])
def test_explicit_cuda_home_precedes_hard_coded_toolkits(
    monkeypatch, tmp_path, torch_cuda_version
):
    explicit = tmp_path / "cuda-12.8"
    explicit.joinpath("bin").mkdir(parents=True)
    explicit.joinpath("bin", "nvcc").touch()
    monkeypatch.setenv("CUDA_HOME", str(explicit))

    find_cuda_home = _load_find_cuda_home(torch_cuda_version)

    assert find_cuda_home() == str(explicit)


def test_v4_fp4_extension_is_guarded_by_sm120_opt_in():
    tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
    sm120_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_enable_sm120" in ast.unparse(node.test)
    ]

    assert any(
        "moe_infinity._v4_fp4" in ast.unparse(node) for node in sm120_guards
    )

    source = SETUP_PATH.read_text(encoding="utf-8")
    assert "compute_120a,code=sm_120a" in source
    assert "if not _v4fp4_arch_flags" not in source
