"""Generator parity (the notebook is generated, not hand-edited), the static validator, citation and notice checks, and
the lock-coverage audit — each with a negative control showing that the check fails when it should."""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from conftest import ROOT, load_tool

build = load_tool("build_notebook")
validator = load_tool("validate_release_assets")
lock_audit = load_tool("audit_lock_coverage")
TEMPLATE = build.load_template(ROOT / "tools" / "notebook_template.py")
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
LOCK = ROOT / TEMPLATE["carried"][TEMPLATE["lock"]]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _carrier(notebook: dict) -> tuple[dict, dict[str, str], dict[str, str]]:
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_sources")]
    assert len(cells) == 1, "exactly one carrier cell is expected"
    values = {}
    for node in ast.parse(_source(cells[0])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CARRIED_FILES", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return cells[0], values["CARRIED_FILES"], values["CARRIED_HASHES"]


def test_carried_files_equal_repository_sources(notebook: dict) -> None:
    _cell, files, _hashes = _carrier(notebook)
    assert list(files) == [*TEMPLATE["carried"], build.SOURCE_RECORD]
    for dest, source in TEMPLATE["carried"].items():
        assert files[dest] == (ROOT / source).read_text(encoding="utf-8").replace("\r\n", "\n"), f"carried {dest} drifted from {source}; regenerate"


def test_carried_hashes_are_correct_and_recorded(notebook: dict) -> None:
    cell, files, hashes = _carrier(notebook)
    assert set(hashes) == set(files)
    for dest, text in files.items():
        assert hashes[dest] == hashlib.sha256(text.encode("utf-8")).hexdigest(), dest
    assert cell["metadata"]["dimer"]["files"] == hashes == notebook["metadata"]["dimer"]["generated_from"]["files"]
    record = json.loads(files[build.SOURCE_RECORD])
    assert record["revision"] == notebook["metadata"]["dimer"]["generated_from"]["revision"]
    assert record["notebook_spec"] == "2.4"


def test_carried_lock_is_the_committed_lock_and_pins_pyproject(notebook: dict) -> None:
    _cell, files, _hashes = _carrier(notebook)
    lock_text = LOCK.read_text(encoding="utf-8")
    assert files[TEMPLATE["lock"]] == lock_text.replace("\r\n", "\n")
    build.check_lock(build._pins(ROOT), lock_text)
    locked = build.lock_packages(lock_text)
    assert locked["torch"] == "2.11.0+cu128" and locked["torchvision"] == "0.26.0+cu128" and locked["bitsandbytes"] == "0.50.2" and locked["diffusers"] == "0.40.0" and locked["transformers"] == "5.16.1"
    assert "--python-platform x86_64-manylinux_2_28" in lock_text.splitlines()[1] and "--only-binary :all:" in lock_text.splitlines()[1]


def test_notebook_metadata_declares_kind_profile_mode(notebook: dict) -> None:
    meta = notebook["metadata"]["dimer"]
    assert (meta["notebook_kind"], meta["notebook_profile"], meta["notebook_mode"], meta["notebook_spec"]) == ("TUTORIAL", "MULTI-CAPABILITY", "GUIDED", "2.4")
    assert [m["revision"] for m in meta["models"]] == ["5c74c11b3ecfb96249b588d17aebd59f3fd9a080", "8d965626ab76734922b00e359ca2d8faab2e0417"]


def test_generator_check_is_clean_and_fails_on_a_hand_edit(notebook: dict, tmp_path: Path) -> None:
    assert build.main(["--check"]) == 0
    edited = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in edited["cells"] if c["cell_type"] == "code" and "run_stage('t2v')" in _source(c))
    cell["source"] = _source(cell) + "\nprint('hand edit')"
    copy = tmp_path / NOTEBOOK.name
    copy.write_bytes(build.to_bytes(edited))
    assert build.main(["--check", "--out", str(copy)]) == 1


def test_kernel_never_imports_the_package(notebook: dict) -> None:
    kernel = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert TEMPLATE["package"] not in kernel
    assert "github.com" not in "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code")


def test_static_validator_passes() -> None:
    assert validator.validate_all() == []


def test_citation_check_fails_on_a_dangling_or_uncited_reference(notebook: dict) -> None:
    markdown = [_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown"]
    assert validator.citation_problems(markdown) == []
    dangling = [*markdown]
    dangling[1] = dangling[1] + " A claim with no entry [13]."
    assert "in-text citation [13] has no reference entry (DOC16)" in validator.citation_problems(dangling)
    uncited = [re.sub(r"\[9\]", "", m) if not m.startswith("## References") else m for m in markdown]
    assert "reference [9] is never cited in the text (DOC16)" in validator.citation_problems(uncited)


def test_notice_check_fails_when_the_canonical_text_is_reworded(notebook: dict) -> None:
    cells = json.loads(json.dumps(notebook["cells"]))
    assert validator.notice_problems(cells) == []
    assert validator.CANONICAL_NOTICE in _source(cells[0]) and _source(cells[0]).startswith("# ")
    cells[0]["source"] = _source(cells[0]).replace("without warranty of any kind", "with limited warranty")
    assert any("DOC21" in p for p in validator.notice_problems(cells))


def test_lock_audit_passes_and_fails_when_a_backend_is_removed(tmp_path: Path) -> None:
    assert lock_audit.audit(LOCK) == []
    text = LOCK.read_text(encoding="utf-8")
    without = re.sub(r"(?ms)^torchvision==.*?(?=^[a-z])", "", text)
    assert "torchvision==" not in without
    copy = tmp_path / "lock.txt"
    copy.write_text(without, encoding="utf-8")
    problems = lock_audit.audit(copy)
    assert any("torchvision" in p for p in problems)


def test_recorded_evidence_is_cited_byte_exact() -> None:
    """Every evidence file cited in docs/release-verification.md exists and matches its cited SHA-256."""
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert validator.EVIDENCE_REF.findall(text), "the recorded Kaggle run must cite its evidence files"
    assert validator.evidence_problems(text) == []


def test_evidence_check_rejects_a_wrong_digest() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    rel, digest = validator.EVIDENCE_REF.findall(text)[0]
    wrong = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    problems = validator.evidence_problems(text.replace(digest, wrong))
    assert any("SHA-256 differs" in p for p in problems)


def test_evidence_check_rejects_a_missing_file() -> None:
    text = "**Status: Candidate**\n`docs/execution-evidence/2099-01-01/missing.ipynb` SHA-256 `" + "0" * 64 + "`\n"
    assert any("missing evidence" in p for p in validator.evidence_problems(text))
