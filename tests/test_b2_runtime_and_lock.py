"""Owner decisions D-2, D-3, D-6 (pass B2): Kaggle-only gate, the CUDA 12.8 lock stack, and the unwaived size gate."""
# ruff: noqa: E501

from __future__ import annotations

import json
import platform
import re

import pytest
from conftest import ROOT, load_tool

build = load_tool("build_notebook")
validator = load_tool("validate_release_assets")
lock_audit = load_tool("audit_lock_coverage")
NOTEBOOK = ROOT / "tutorials" / "kandinsky5_video_generation_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-kaggle.lock.txt"


def _check_cell() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def test_lock_is_the_cuda_12_8_stack() -> None:
    text = LOCK.read_text(encoding="utf-8")
    assert lock_audit.cuda_stack_problems(text) == []
    assert validator.lock_problems() == []
    names = lock_audit.lock_packages(text)
    nvidia = [n for n in names if n.startswith("nvidia-")]
    assert nvidia and all(n.endswith("-cu12") for n in nvidia)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("torch==2.11.0+cu128", "torch==2.11.0", "not a +cu128 build"),
        ("torchvision==0.26.0+cu128", "torchvision==0.26.0", "not a +cu128 build"),
        ("nvidia-nvjitlink-cu12==12.8.93", "nvidia-nvjitlink==13.0.88", "not a CUDA 12 (-cu12) NVIDIA wheel"),
        ("nvidia-cublas-cu12==12.8.4.1", "nvidia-cublas-cu13==13.1.0.3", "not part of the CUDA 12.8 stack"),
    ],
)
def test_lock_check_fails_on_a_cuda_13_or_non_cu128_entry(old: str, new: str, message: str) -> None:
    text = LOCK.read_text(encoding="utf-8")
    assert old in text
    problems = lock_audit.cuda_stack_problems(text.replace(old, new))
    assert any(message in p for p in problems), problems


def test_install_cell_reaches_only_pypi_and_the_cu128_index() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "'--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match'" in code
    assert "'--require-hashes'" in code
    pattern = dict(validator.FORBIDDEN_IN_CODE)["extra index other than the PyTorch cu128 index (ENV11)"]
    assert not pattern.search(code)
    assert pattern.search(code.replace("https://download.pytorch.org/whl/cu128", "https://example.org/simple"))


def test_check_cell_refuses_a_non_kaggle_linux_session(monkeypatch) -> None:
    """With Linux x86_64 faked and no Kaggle markers, the first cell stops before probing the GPU (D-2)."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    with pytest.raises(RuntimeError, match="This is not a Kaggle notebook session"):
        exec(compile(_check_cell(), "check_cell", "exec"), {})


def test_check_cell_requires_both_markers(monkeypatch) -> None:
    """The env var alone is not enough on a host without /kaggle (this Windows host has none)."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    with pytest.raises(RuntimeError, match="'/kaggle': False"):
        exec(compile(_check_cell(), "check_cell", "exec"), {})


def test_size_gate_is_recorded_open_and_not_waived() -> None:
    for doc in ("STATUS.md", "docs/WEIGHTS.md", "docs/release-verification.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert re.search(r"not waived", text), doc
        assert "gate is waived" not in text.replace("not waived", "")


def test_probe_logs_are_committed_byte_for_byte() -> None:
    import hashlib

    expected = {
        "kandinsky5-lite-t4-smoke_v1.log": "5bae7d4f69075b023e15ebde609017876a0a9334f0456e7505996af405fa4c48",
        "kandinsky5-lite-t4-smoke_v2.log": "184e78148b9aa0a1fbdf9f73012c356f0137a75418500118f02aee5ff6831248",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "docs" / "development-probe" / name).read_bytes()).hexdigest() == digest
    assert "docs/development-probe/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
