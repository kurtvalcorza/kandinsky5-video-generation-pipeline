"""V-F2 (pass B3): the two-phase single-index install and the index-coverage check, offline."""
# ruff: noqa: E501
from __future__ import annotations

import ast
import json

from conftest import ROOT, load_tool

build = load_tool("build_notebook")
coverage = load_tool("check_index_coverage")
lock_audit = load_tool("audit_lock_coverage")
NOTEBOOK = ROOT / "tutorials" / "kandinsky5_video_generation_colab.ipynb"
LOCK_TEXT = (ROOT / "tutorials" / "requirements-kaggle.lock.txt").read_text(encoding="utf-8")


def _notebook_split_lock():
    """The split_lock function exactly as the notebook carries it, compiled from the install cell."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c["source"] for c in nb["cells"] if c["cell_type"] == "code" and "def split_lock(" in c["source"])
    function = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "split_lock")
    namespace: dict = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "install_cell", "exec"), namespace)
    return namespace["split_lock"]


def test_notebook_split_matches_the_tool_and_covers_the_lock_once() -> None:
    carried = _notebook_split_lock()(LOCK_TEXT, list(build.PHASE_A_PACKAGES))
    assert carried == build.split_lock(LOCK_TEXT, build.PHASE_A_PACKAGES)
    phase_a, phase_b = (build.lock_packages(t) for t in carried)
    assert phase_a == {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}
    assert len(phase_b) == 64 and not set(phase_a) & set(phase_b)
    assert carried[0].count("--hash=sha256:") + carried[1].count("--hash=sha256:") == LOCK_TEXT.count("--hash=sha256:")
    assert lock_audit.install_split_problems(LOCK_TEXT) == []


def test_audit_rejects_a_cu128_build_that_would_land_in_the_pypi_phase() -> None:
    bad = LOCK_TEXT.replace("triton==3.6.0 ", "triton==3.6.0+cu128 ")
    assert "triton==3.6.0+cu128" in bad
    assert any("phase B (PyPI) contains a +cu128 build" in p for p in lock_audit.install_split_problems(bad))


SYNTHETIC_LOCK = """triton==3.6.0 \\
    --hash=sha256:{pypi}
torch==2.11.0+cu128 \\
    --hash=sha256:{torch}
""".format(pypi="a" * 64, torch="b" * 64)
WHEEL = "triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"


def test_coverage_split_passes_and_mixed_reports_the_triton_hazard() -> None:
    def pypi(name, version):
        return {"triton": [(WHEEL, "a" * 64)], "torch": []}[name]

    def torch_index(name, version):
        return {"triton": [(WHEEL, "c" * 64)], "torch": [("torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl", "b" * 64)]}[name]

    lines, problems = coverage.check(SYNTHETIC_LOCK, "split", pypi, torch_index)
    assert problems == [] and len(lines) == 2
    _lines, problems = coverage.check(SYNTHETIC_LOCK, "mixed", pypi, torch_index)
    assert len(problems) == 1 and problems[0].startswith("HAZARD triton==3.6.0: cu128 serves installable") and "c" * 64 in problems[0]


def test_coverage_split_fails_when_the_phase_index_lacks_a_locked_hash() -> None:
    _lines, problems = coverage.check(SYNTHETIC_LOCK, "split", lambda n, v: [(WHEEL, "d" * 64)], lambda n, v: [("torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl", "b" * 64)])
    assert problems == ["triton==3.6.0: index pypi serves no installable wheel with a locked sha256"]


def test_wheel_tag_filter() -> None:
    ok = coverage.wheel_installable
    assert ok(WHEEL) and ok("numpy-2.5.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl") and ok("six-1.0-py3-none-any.whl")
    assert ok("cffi-1.0-cp39-abi3-manylinux2014_x86_64.whl")
    assert not ok("numpy-2.5.3-cp313-cp313-manylinux_2_28_x86_64.whl")
    assert not ok("numpy-2.5.3-cp312-cp312-manylinux_2_34_x86_64.whl")
    assert not ok("numpy-2.5.3-cp312-cp312-manylinux_2_28_aarch64.whl")
    assert not ok("numpy-2.5.3.tar.gz")
