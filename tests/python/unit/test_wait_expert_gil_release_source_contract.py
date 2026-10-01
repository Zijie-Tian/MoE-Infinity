from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_wait_expert_releases_gil_while_native_workers_finish() -> None:
    source = (ROOT / "core/python/py_archer_prefetch.cpp").read_text(
        encoding="utf-8"
    )
    start = source.index('.def("wait_expert"')
    end = source.index('.def("dispatch_experts"', start)
    binding = source[start:end]
    assert "&ExpertDispatcher::WaitHiddenStates" in binding
    assert "py::call_guard<py::gil_scoped_release>()" in binding


def test_enqueue_expert_releases_gil_while_eviction_claim_finishes() -> None:
    source = (ROOT / "core/python/py_archer_prefetch.cpp").read_text(
        encoding="utf-8"
    )
    start = source.index('.def("enqueue_expert"')
    end = source.index('.def("set_inputs"', start)
    binding = source[start:end]
    assert "&ExpertDispatcher::EnqueueExpert" in binding
    assert "py::call_guard<py::gil_scoped_release>()" in binding
