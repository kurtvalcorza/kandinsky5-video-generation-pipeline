#!/usr/bin/env python3
"""Static release-asset validator for the Kandinsky 5.0 Video Lite tutorial and model card (NOTEBOOK_SPEC 2.4).

Checks the committed notebook's structure, metadata, guided layer, write-up layer (problem statement, data context,
what did not work, references, use notice), citation correspondence (DOC16), forbidden patterns, form-field defaults,
the model card's required sections, and the release status. It is a static check: it is not execution evidence (REL8).

Usage:
    python tools/validate_release_assets.py            # exit 1 on the first class of failures, listing each
    python tools/validate_release_assets.py --notebook PATH
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "kandinsky5_video_pipeline"
REPO_NAME = "kandinsky5-video-generation-pipeline"
NOTEBOOK = ROOT / "tutorials" / "kandinsky5_video_generation_colab.ipynb"
NOTEBOOK_SPEC = "2.4"
EXPECTED_METADATA = {"notebook_kind": "TUTORIAL", "notebook_profile": "MULTI-CAPABILITY", "notebook_mode": "GUIDED", "notebook_spec": NOTEBOOK_SPEC}
TUTORIAL_FILENAME = re.compile(r"^[a-z0-9]+_[a-z0-9_]+_colab\.ipynb$")  # KND2: <model>_<task>_colab.ipynb

CANONICAL_NOTICE = (
    "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
    "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
    "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
    "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
    "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
    "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
    "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
    "> its use."
)

REQUIRED_HEADINGS = (
    "## Problem statement",
    "## Data context",
    "## How to use this notebook",
    "## The task: Input → Model/System → Output",
    "## Roadmap",
    "## Glossary",
    "## Prerequisites",
    "## 1. Check the runtime",
    "## 2. Carry the code and install the locked runtime",
    "## 3. Pin, stage and verify the models",
    "## 4. Inputs, validation and refusals",
    "## 5. Encode the prompts once, then release the encoder",
    "## 6. Capability A — text-to-video",
    "## 7. Capability B — image-to-video",
    "## 8. Evaluate both clips",
    "## 9. Export and provenance",
    "## 10. Optional activity",
    "## 11. Bring Your Own Data (optional)",
    "## Interpretation and limits",
    "## What did not work",
    "## Troubleshooting",
    "## References",
)
GUIDED_MIN_COUNTS = {"Predict before running": 5, "What to notice": 5, "Check your reasoning": 5, "Expected result": 2}
SECTION_TAGS = ("[Concept]", "[Evaluation practice]", "[Engineering]")
STAGE_CALLS = ("run_stage('weights')", "run_stage('prepare')", "run_stage('encode')", "run_stage('t2v')", "run_stage('i2v')", "run_stage('evaluate')", "run_stage('report')", "run_stage('activity')")
FORM_FIELDS = (
    'RUN_ACTIVITY = False  # @param {type:"boolean"}',
    'USE_BYOD = False  # @param {type:"boolean"}',
    "BYOD_IMAGE_PATH = ''  # @param {type:\"string\"}",
    "BYOD_T2V_PROMPT = ''  # @param {type:\"string\"}",
    "BYOD_I2V_PROMPT = ''  # @param {type:\"string\"}",
)
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|WRITE ME|Insert text here|Tooltip:", re.I)
PROMOTIONAL = re.compile(r"\b(simply|just|easily|obviously)\b", re.I)
FORBIDDEN_IN_CODE = (
    ("repository clone (ST3)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install (ST3)", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("mutable model reference (MOD2)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled (MOD6)", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    ("unsafe deserialization (MOD9)", re.compile(r"\bpickle\.load|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)|weights_only\s*=\s*False")),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape (SRC6)", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
    ("unhashed or source install (ENV11)", re.compile(r"--no-binary|--no-build-isolation|--trusted-host")),
    ("extra index other than the PyTorch cu128 index (ENV11)", re.compile(r"--extra-index-url(?!', 'https://download\.pytorch\.org/whl/cu128')")),
    ("broad warning suppression (SRC11)", re.compile(r"filterwarnings\(\s*['\"]ignore['\"]\s*\)")),
    ("developer-local path (SRC7)", re.compile(r"[A-Za-z]:\\\\Users|/home/[a-z]|/Users/[A-Za-z]")),
    ("credential in source (OUT10)", re.compile(r"hf_[A-Za-z0-9]{30,}|HF_TOKEN\s*=")),
)
KERNEL_PACKAGE_IMPORT = re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"), (4, "Intended Use and Limitations"), (6, "Primary Intended Uses"), (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"), (4, "Factors"), (6, "Groups"), (6, "Instrumentation"), (6, "Environment"), (4, "Metrics"),
    (6, "Performance Measures"), (6, "Decision thresholds"), (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"), (6, "Data"), (6, "Human Life"), (6, "Mitigations"), (6, "Risks and harms"), (6, "Use cases"),
]
CARD_FRONT_MATTER = ("license:", 'model_card_spec: "1.2"', "pipeline_tag:", "base_model:", "date_published:")
UNSUPPORTED_CLAIMS = re.compile(r"\b(production[- ]ready|state[- ]of[- ]the[- ]art results|is release-grade|now release-grade)\b", re.I)


class ValidationError(AssertionError):
    """Raised with every defect found; each line names the file and the rule."""


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def citation_problems(markdown_cells: list[str]) -> list[str]:
    """DOC14–DOC16: every in-text [n] resolves to a reference entry, every entry is cited, every entry has a link."""
    refs_cells = [c for c in markdown_cells if c.lstrip().startswith("## References")]
    if len(refs_cells) != 1:
        return [f"expected exactly one '## References' cell, found {len(refs_cells)}"]
    refs = refs_cells[0]
    if markdown_cells[-1] is not refs:
        return ["the References cell must be the last markdown cell (DOC13)"]
    entries = {int(m.group(1)): m.group(2) for m in re.finditer(r"^\[(\d+)\] (.+)$", refs, re.M)}
    problems = []
    if sorted(entries) != list(range(1, len(entries) + 1)):
        problems.append(f"reference numbers are not 1..n in order: {sorted(entries)}")
    for number, text in entries.items():
        if not re.search(r"https://\S+", text):
            problems.append(f"reference [{number}] has no resolvable https link (DOC15)")
        if "[Online]. Available:" not in text:
            problems.append(f"reference [{number}] does not follow the IEEE style used throughout (DOC14)")
    body = "\n".join(c for c in markdown_cells if c is not refs)
    body = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", body)  # badges
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()", body) for n in re.findall(r"\d+", group)}
    for number in sorted(cited - set(entries)):
        problems.append(f"in-text citation [{number}] has no reference entry (DOC16)")
    for number in sorted(set(entries) - cited):
        problems.append(f"reference [{number}] is never cited in the text (DOC16)")
    return problems


def notice_problems(cells: list[dict]) -> list[str]:
    """DOC17–DOC21: the canonical notice (with the BYOD sentence) sits after the title and before the first code cell."""
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    holders = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and CANONICAL_NOTICE in _source(c)]
    problems = []
    if not holders:
        problems.append("the canonical use notice text is missing or reworded (DOC21)")
    elif holders[0] > first_code:
        problems.append("the use notice must come before the first code cell (DOC17)")
    elif not _source(cells[holders[0]]).lstrip().startswith("# "):
        problems.append("the use notice must follow the title (DOC17)")
    if any("[!WARNING]" in _source(c) or "[!NOTE]" in _source(c) for c in cells):
        problems.append("GitHub alert syntax is not rendered by hosted runtimes; use a plain blockquote (DOC20)")
    return problems


def notebook_problems(path: Path) -> list[str]:
    problems: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    meta = notebook["metadata"].get("dimer", {})
    if notebook.get("nbformat") != 4:
        problems.append("nbformat must be 4 (SRC1)")
    if not TUTORIAL_FILENAME.match(path.name):
        problems.append(f"{path.name} does not follow the TUTORIAL filename pattern <model>_<task>_colab.ipynb (KND2)")
    for key, value in EXPECTED_METADATA.items():
        if meta.get(key) != value:
            problems.append(f"metadata.dimer.{key} is {meta.get(key)!r}, expected {value!r} (KND1, §3.4)")
    if meta.get("standalone") is not True or meta.get("requires_dimer_worker") is not False:
        problems.append("metadata must declare standalone=True and requires_dimer_worker=False (ST2)")
    ids = [c.get("id") for c in cells]
    if len(set(ids)) != len(ids) or not all(ids):
        problems.append("cell ids must be present and unique (SRC1)")
    markdown_cells = [_source(c) for c in cells if c["cell_type"] == "markdown"]
    markdown = "\n".join(markdown_cells)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append(f"cell {cell['id']} has persisted outputs or an execution count (SRC4)")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"cell {cell['id']} does not compile: {exc} (SRC9)")
        for name, pattern in FORBIDDEN_IN_CODE:
            if pattern.search(source):
                problems.append(f"cell {cell['id']}: {name}")
        if not cell["metadata"].get("dimer", {}).get("embedded_sources") and KERNEL_PACKAGE_IMPORT.search(source):
            problems.append(f"cell {cell['id']}: the kernel imports the repository package (ST3); stages import the carried copy")
        if source.startswith("# @title Infrastructure:") and cell["metadata"].get("cellView") != "form":
            problems.append(f"cell {cell['id']}: infrastructure cell is not collapsed (GDL11)")
    infrastructure = sum(_source(c).startswith("# @title Infrastructure:") for c in cells if c["cell_type"] == "code")
    if infrastructure != 4:
        problems.append(f"expected 4 collapsed infrastructure cells, found {infrastructure} (GDL11)")
    code = "\n".join(_source(c) for c in cells if c["cell_type"] == "code")
    for call in STAGE_CALLS:
        if call not in code:
            problems.append(f"missing learner stage call {call}")
    for field in FORM_FIELDS:
        if field not in code:
            problems.append(f"missing or non-default form field: {field} (EXE1, RUN2, RUN9)")
    position = 0
    for heading in REQUIRED_HEADINGS:
        found = markdown.find(heading, position)
        if found < 0:
            problems.append(f"missing or out-of-order heading: {heading}")
        else:
            position = found
    for marker, minimum in GUIDED_MIN_COUNTS.items():
        if markdown.count(marker) < minimum:
            problems.append(f"guided layer: '{marker}' appears {markdown.count(marker)} times, expected >= {minimum} (GDL7–GDL9)")
    for tag in SECTION_TAGS:
        if tag not in markdown:
            problems.append(f"section tag {tag} is never used (GDL12)")
    for phrase in ("**This notebook runs on Kaggle only.**", "KAGGLE_KERNEL_RUN_TYPE", "uncalibrated tutorial evidence", "docs/development-probe/", "Predict → Change one thing → Run → Observe → Explain", "**Conclusion template.**", "**Transfer.**", "development-run evidence", "Do not upload confidential or restricted", "Kaggle", "**Duration (estimate):**"):
        if phrase not in markdown:
            problems.append(f"missing required phrase: {phrase!r}")
    if PLACEHOLDER.search(markdown + code):
        problems.append(f"placeholder text survives: {PLACEHOLDER.search(markdown + code).group(0)!r} (SRC3)")
    learner_prose = "\n".join(m for m in markdown_cells if "## References" not in m)
    if PROMOTIONAL.search(learner_prose):
        problems.append(f"learner prose uses {PROMOTIONAL.search(learner_prose).group(0)!r} (WRT8)")
    problems += notice_problems(cells)
    problems += citation_problems(markdown_cells)
    return problems


def lock_problems(lock: Path = ROOT / "tutorials" / "requirements-kaggle.lock.txt") -> list[str]:
    """ENV11 + owner decision D-3: the carried lock is the CUDA 12.8 stack (+cu128 torch, -cu12 NVIDIA wheels only)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_lock_coverage", ROOT / "tools" / "audit_lock_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [f"{lock.name}: {problem}" for problem in module.cuda_stack_problems(lock.read_text(encoding="utf-8"))]


def card_problems() -> list[str]:
    problems = []
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    front = card.split("---", 2)
    if not card.startswith("---") or len(front) < 3:
        return ["MODEL_CARD.md must open with a YAML front-matter block (G1)"]
    for key in CARD_FRONT_MATTER:
        if key not in front[1]:
            problems.append(f"MODEL_CARD.md front matter lacks {key} (G1)")
    if len(re.findall(r"^# ", card, re.M)) != 1:
        problems.append("MODEL_CARD.md must have exactly one level-1 heading (G3)")
    position = 0
    for level, title in REQUIRED_CARD_HEADINGS:
        m = re.compile(rf"^{'#' * level} {re.escape(title)}\s*$", re.M | re.I).search(card, position)
        if not m:
            problems.append(f"MODEL_CARD.md: missing or out-of-order {'#' * level} {title} (G4–G6)")
        else:
            position = m.end()
    if PLACEHOLDER.search(card):
        problems.append("MODEL_CARD.md contains placeholder text (G9)")
    for doc in ("MODEL_CARD.md", "README.md", "STATUS.md", "docs/release-verification.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        if UNSUPPORTED_CLAIMS.search(text):
            problems.append(f"{doc}: unsupported claim {UNSUPPORTED_CLAIMS.search(text).group(0)!r}")
    from_pipeline = (ROOT / "src" / PACKAGE / "pipeline.py").read_text(encoding="utf-8")
    for name in ("MODEL_REVISION", "I2V_REVISION"):
        revision = re.search(rf'^{name} = "([0-9a-f]{{40}})"$', from_pipeline, re.M).group(1)
        for doc in ("MODEL_CARD.md", "README.md"):
            if revision not in (ROOT / doc).read_text(encoding="utf-8"):
                problems.append(f"{doc} does not state {name} {revision}")
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    if "**Candidate**" not in status:
        problems.append("STATUS.md must state the status **Candidate** until execution evidence for this blob is recorded (REL14)")
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for doc in ("STATUS.md", "docs/WEIGHTS.md"):
        if "not waived" not in (ROOT / doc).read_text(encoding="utf-8"):
            problems.append(f"{doc} must record the 9 GB size gate as open and not waived (owner decision D-6)")
    if "Candidate — no recorded execution" not in verification:
        problems.append("docs/release-verification.md must state 'Candidate — no recorded execution' (REL1, REL14)")
    return problems


def validate_all(notebook: Path = NOTEBOOK) -> list[str]:
    return notebook_problems(notebook) + lock_problems() + card_problems()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args(argv)
    problems = validate_all(args.notebook)
    if problems:
        for problem in problems:
            print("FAIL:", problem, file=sys.stderr)
        return 1
    print(f"OK: {args.notebook.name}, MODEL_CARD.md, STATUS.md and docs/release-verification.md pass the static checks (not execution evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
