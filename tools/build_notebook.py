#!/usr/bin/env python3
"""Generate the STANDALONE DIMER tutorial notebook (NOTEBOOK_SPEC 2.4 §4, §8.1) from repository sources.

Adapted from the generator of the sibling ``kandinsky-generation-pipeline`` (build_notebook.py/3.0). Nothing is
pip-installed into the notebook kernel: the notebook

1. checks the runtime (Linux x86_64, a T4-class CUDA GPU, host RAM, disk) and creates a fresh run directory ``ROOT``;
2. writes the carried files — the repository's package under ``src/``, the stage runner, the hash-locked
   requirements, the two pinned snapshot manifests, the licence and ``source.json`` — to ``ROOT`` and verifies each
   against ``CARRIED_HASHES`` (the carried text stays visible in that cell: ST5, SRC12);
3. downloads a pinned ``uv`` wheel (URL + size + SHA-256), builds a managed-Python virtual environment, installs the
   lock with ``--require-hashes --only-binary :all:`` into it, and defines ``run_stage``, which runs one stage of the
   carried runner per process and re-raises a failed stage's error message in the kernel;
4. runs the template's learner cells, which call ``run_stage(...)`` and display the files it writes.

Usage (from the repository root, or with --repo):
    python tools/build_notebook.py            # write tutorials/<notebook_name>
    python tools/build_notebook.py --check    # exit 1 if the committed notebook differs from the generator output
    python tools/build_notebook.py --out PATH # write elsewhere (review copies)
"""
# ruff: noqa: E501  -- learner-facing prose and generated code are kept on single lines so they render readably
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "build_notebook.py/3.3-k5video"
NOTEBOOK_SPEC = "2.4"
SOURCE_RECORD = "source.json"
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY", "I2V_ID", "I2V_REVISION", "I2V_KEY")
REQUIRED_KEYS = (
    "package", "repo_name", "stem", "notebook_name", "profile", "mode", "kind", "run_all", "byod", "title", "badges",
    "capability", "notice", "intro", "learning_objectives", "exclusions", "prerequisites", "carried", "stage_runner", "lock",
    "managed_python", "uv", "torch_index", "runtime_gate", "runtime_modules", "setup", "cells", "closing", "guided",
)
PROFILES = ("E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY")
MODES = ("REFERENCE", "GUIDED", "WORKSHOP")
KINDS = ("TUTORIAL", "DEMO", "WORKSHOP", "CAPSTONE")


def load_template(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("notebook_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.TEMPLATE
    missing = [k for k in REQUIRED_KEYS if k not in template]
    if missing:
        raise SystemExit(f"template missing keys: {missing}")
    return template


def _pins(repo: Path) -> list[str]:
    """The `==` runtime pins of pyproject.toml (ENV2); the lock is compiled from exactly these."""
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not block:
        raise SystemExit("pyproject.toml: dependencies block not found")
    pins = re.findall(r'"([^"]+)"', block.group(1))
    bad = [p for p in pins if "==" not in p]
    if bad:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {bad}")
    return pins


def lock_packages(lock_text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock."""
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


PHASE_A_PACKAGES = ("torch", "torchvision")


def split_lock(lock_text: str, phase_a: tuple[str, ...] | list[str] = ("torch", "torchvision")) -> tuple[str, str]:
    """Split a uv hash lock into (phase A, phase B) requirement texts; every entry keeps its hashes, each lands once."""
    import re as _re

    first, second = [], []
    for block in _re.split(r"\n(?=[A-Za-z0-9])", lock_text):
        match = _re.match(r"([A-Za-z0-9._-]+)==", block)
        if match:
            (first if match.group(1).lower() in phase_a else second).append(block.rstrip("\n"))
    return "\n".join(first) + "\n", "\n".join(second) + "\n"


def check_lock(pins: list[str], lock_text: str) -> None:
    """Every direct pin must appear in the lock at the same version, and every lock entry must carry a hash."""
    locked = lock_packages(lock_text)
    for pin in pins:
        name, version = pin.split("==", 1)
        if locked.get(name.lower()) != version:
            raise SystemExit(f"lock does not pin {pin} (found {locked.get(name.lower())}); recompile the lock")
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock_text.split("\n", 2)[-1])
    unhashed = [b.split("==", 1)[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")


def _head_revision(repo: Path) -> str:
    """HEAD at generation time: a provenance label only; parity is anchored on file content (see ``--check``)."""
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or "uncommitted"


def recorded_revision(notebook_path: Path) -> str | None:
    """`generated_from.revision` of an existing notebook, or None."""
    if not notebook_path.exists():
        return None
    try:
        meta = json.loads(notebook_path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
        return str(meta["revision"])
    except (KeyError, ValueError, TypeError):
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def carried_files(repo: Path, template: dict[str, Any], revision: str) -> dict[str, str]:
    """`{destination: text}` of every carried file (LF newlines), plus the generated `source.json` record last."""
    files: dict[str, str] = {}
    for dest, source in template["carried"].items():
        path = repo / source
        if not path.is_file():
            raise SystemExit(f"carried source missing: {source}")
        files[dest] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for key in (template["stage_runner"], template["lock"]):
        if key not in files:
            raise SystemExit(f"template names {key!r} but does not carry it")
    check_lock(_pins(repo), files[template["lock"]])
    record = {
        "repository": f"kurtvalcorza/{template['repo_name']}",
        "revision": revision,
        "generator": GENERATOR_VERSION,
        "notebook_spec": NOTEBOOK_SPEC,
        "files": {dest: sha256_text(text) for dest, text in files.items()},
        "sources": dict(template["carried"]),
    }
    files[SOURCE_RECORD] = json.dumps(record, indent=2) + "\n"
    return files


def load_context(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    pkg = template["package"]
    revision = revision or _head_revision(repo)
    files = carried_files(repo, template, revision)
    modules = [d for d in template["carried"] if d.startswith(f"src/{pkg}/") and d.endswith(".py")]
    pipeline = files[f"src/{pkg}/pipeline.py"]
    ident: dict[str, str] = {}
    for name in IDENTITY_NAMES:
        m = re.search(rf'^{name} = "([^"]+)"$', pipeline, re.M)
        if not m:
            raise SystemExit(f"pipeline.py: {name} not found as a top-level string constant")
        ident[name] = m.group(1)
    manifests = {d: json.loads(t) for d, t in files.items() if d.startswith("weights/") and d.endswith("dimer-base-manifest.json")}
    for key, model_id, revision_pin in ((ident["MODEL_KEY"], ident["MODEL_ID"], ident["MODEL_REVISION"]), (ident["I2V_KEY"], ident["I2V_ID"], ident["I2V_REVISION"])):
        manifest = manifests.get(f"weights/{key}/dimer-base-manifest.json")
        if manifest is None:
            raise SystemExit(f"the {key} snapshot manifest is not carried")
        if (manifest["modelId"], manifest["revision"]) != (model_id, revision_pin):
            raise SystemExit(f"{key}: manifest identity != module identity")
    return {
        "pkg": pkg,
        "revision": revision,
        "files": files,
        "hashes": {d: sha256_text(t) for d, t in files.items()},
        "modules": modules,
        "module_sha256": hashlib.sha256("".join(files[m] for m in modules).encode("utf-8")).hexdigest(),
        "manifests": manifests,
        "total_gb": sum(m["totalBytes"] for m in manifests.values()) / 1e9,
        "lock_packages": len(lock_packages(files[template["lock"]])),
        **ident,
    }


def _md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": "", "metadata": metadata or {}, "outputs": [], "source": source.rstrip("\n")}


# GDL11: collapsed-by-default metadata for infrastructure cells (Colab form view; Jupyter source_hidden).
INFRASTRUCTURE_METADATA: dict[str, Any] = {"cellView": "form", "jupyter": {"source_hidden": True}}


# ---- infrastructure cell sources -------------------------------------------------------------------------------------

CHECK_CELL = """# @title Infrastructure: check the runtime (Kaggle T4), GPU, host RAM and disk; create a fresh run directory
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

SESSION_START = time.perf_counter()
RUNTIME_NOTE = 'This notebook is supported on a Kaggle notebook with a T4 GPU (Settings > Accelerator > GPU T4 x2, Internet on). See Prerequisites and Troubleshooting.'
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Unsupported platform: the locked environment is built for Linux x86_64 (manylinux). ' + RUNTIME_NOTE)
# Kaggle-only (owner decision): a Kaggle notebook sets KAGGLE_KERNEL_RUN_TYPE to 'Interactive' or 'Batch' and mounts /kaggle.
# Both markers are required, so a large non-Kaggle session (for example Colab Pro High-RAM) cannot pass the resource gate untested.
KAGGLE_MARKERS = {{'KAGGLE_KERNEL_RUN_TYPE': os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''), '/kaggle': Path('/kaggle').is_dir()}}
if KAGGLE_MARKERS['KAGGLE_KERNEL_RUN_TYPE'] not in ('Interactive', 'Batch') or not KAGGLE_MARKERS['/kaggle']:
    raise RuntimeError(f'This is not a Kaggle notebook session (markers found: {{KAGGLE_MARKERS}}). The notebook is tested and supported on Kaggle only; open it in Kaggle (File > Import Notebook), then Run all. ' + RUNTIME_NOTE)
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('No NVIDIA GPU driver was found. ' + RUNTIME_NOTE) from exc
if gpu.returncode or not gpu.stdout.strip():
    raise RuntimeError('CUDA was not detected. ' + RUNTIME_NOTE)
gpu_name, gpu_mib, gpu_cc = [field.strip() for field in gpu.stdout.strip().splitlines()[0].split(',')]
if float(gpu_mib) < {min_gpu_mib} or float(gpu_cc) < {min_cc}:
    raise RuntimeError(f'GPU {{gpu_name}} ({{gpu_mib}} MiB, compute capability {{gpu_cc}}) is below the T4 class this notebook needs (>= {min_gpu_mib} MiB, >= {min_cc}). ' + RUNTIME_NOTE)
meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
ram_total_gb = int(meminfo['MemTotal'].split()[0]) * 1024 / 1e9
ram_available_gb = int(meminfo['MemAvailable'].split()[0]) * 1024 / 1e9
if ram_available_gb < {min_ram_gb}:
    raise RuntimeError(f'Only {{ram_available_gb:.1f}} GB of host RAM is available (total {{ram_total_gb:.1f}} GB). Loading the 4-bit text encoder peaked at about {peak_ram_gb} GB of host memory in development runs, so this notebook needs at least {min_ram_gb} GB available. Free Colab (about 12.7 GB) is not enough. ' + RUNTIME_NOTE)
STEM = {stem!r}
ROOT = Path.cwd() / 'outputs' / STEM / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
# The weights go to the temporary disk, not the working directory: Kaggle caps /kaggle/working at about 20 GB.
WEIGHTS = Path(tempfile.gettempdir()) / (STEM + '_weights')
WEIGHTS.mkdir(exist_ok=True)
ENV_ROOT = Path(tempfile.gettempdir()) / (STEM + '_env_' + ROOT.name)
staged_gb = sum(p.stat().st_size for p in WEIGHTS.rglob('*') if p.is_file()) / 1e9
need_gb = max(0.0, {weights_gb} - staged_gb) + {env_gb}
free_gb = shutil.disk_usage(WEIGHTS).free / 1e9
if free_gb < need_gb:
    raise RuntimeError(f'Not enough free disk: about {{need_gb:.0f}} GB is needed for the weights and the environment, {{free_gb:.0f}} GB is free. Start a fresh session (see Troubleshooting).')
print({{'gpu': gpu_name, 'gpu_mib': int(float(gpu_mib)), 'compute_capability': gpu_cc, 'host_ram_available_gb': round(ram_available_gb, 1), 'kaggle': Path('/kaggle').is_dir(), 'kernel_python': platform.python_version(), 'run_directory': str(ROOT), 'weights': str(WEIGHTS), 'free_disk_gb': round(free_gb)}})"""

CARRIER_CELL = """# @title Infrastructure: write and verify the carried package, stage runner, lock and manifests
CARRIED_FILES = {files}
CARRIED_HASHES = {hashes}
for name, text in CARRIED_FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\\n')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name + '. Do not edit this cell; regenerate the notebook from the repository.')
NOTEBOOK_SOURCE = json.loads((ROOT / {source!r}).read_text(encoding='utf-8'))
print({{'carried_files': len(CARRIED_FILES), 'verified': True, 'repository': NOTEBOOK_SOURCE['repository'], 'revision': NOTEBOOK_SOURCE['revision'], 'generator': NOTEBOOK_SOURCE['generator']}})"""

INSTALL_CELL = """# @title Infrastructure: install the locked runtime into an isolated environment and define the stage runner
import io
import urllib.error
import urllib.request
import zipfile

from IPython.display import Image, Video, display

UV_URL = {uv_url!r}
UV_BYTES = {uv_bytes}
UV_SHA256 = {uv_sha256!r}
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(UV_BYTES + 1)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv {uv_version} wheel size/hash mismatch: refusing to run it')
ENV_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ENV_ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
# Stage processes get no Hugging Face token (every download is public), no kernel Python path, and only the first GPU,
# so a two-GPU Kaggle node behaves like the single T4 the notebook was sized for.
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', UV_CACHE_DIR=str(ENV_ROOT / 'cache'), MPLBACKEND='Agg')
for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME'):
    ENV.pop(name, None)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', {python!r}, str(ENV_ROOT / 'venv')], env=ENV, check=True)
PYTHON = ENV_ROOT / 'venv' / 'bin' / 'python'
# Two-phase install from the one carried lock, each phase from exactly one index (no cross-index choice):
# phase A takes torch and torchvision (+cu128 builds) from the PyTorch CUDA 12.8 index, phase B takes every other entry
# from PyPI. Both phases use --no-deps --require-hashes, so nothing outside the lock is installed, and `uv pip check`
# then confirms the combined environment is consistent. (A single mixed-index install pulled triton from the PyTorch
# mirror, whose wheel differs from the PyPI wheel the lock hashes.)
{split_source}

PHASE_A, PHASE_B = split_lock((ROOT / {lock!r}).read_text(encoding='utf-8'), {phase_a!r})
(ROOT / 'requirements-phase-a-pytorch.txt').write_text(PHASE_A, encoding='utf-8')
(ROOT / 'requirements-phase-b-pypi.txt').write_text(PHASE_B, encoding='utf-8')
for index_url, requirements in (({torch_index!r}, 'requirements-phase-a-pytorch.txt'), ('https://pypi.org/simple', 'requirements-phase-b-pypi.txt')):
    subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--no-deps', '--require-hashes', '--only-binary', ':all:', '--index-url', index_url, '-r', str(ROOT / requirements)], env=ENV, check=True)
subprocess.run([str(UV), 'pip', 'check', '--python', str(PYTHON)], env=ENV, check=True)
probe = subprocess.run([str(PYTHON), '-c', {probe!r}], env=ENV, check=True, capture_output=True, text=True)
RUNTIME = json.loads(probe.stdout.strip().splitlines()[-1])
print({{'notebook_source': NOTEBOOK_SOURCE['revision'], **RUNTIME, 'locked_packages': {n_locked}, 'environment': str(ENV_ROOT / 'venv'), 'setup_seconds': round(time.perf_counter() - SESSION_START)}})
if not RUNTIME['cuda']:
    raise RuntimeError('The isolated environment cannot see a CUDA GPU. See Troubleshooting: this notebook does not support CPU-only runtimes.')


def run_stage(stage, *options):
    \"\"\"Run one stage of the carried runner in its own process with the isolated interpreter; stream its output.\"\"\"
    log = ROOT / 'logs' / (stage + '.log')
    log.parent.mkdir(exist_ok=True)
    error_file = ROOT / 'state' / (stage + '.error.json')
    error_file.unlink(missing_ok=True)
    command = [str(PYTHON), '-u', str(ROOT / {runner!r}), '--root', str(ROOT), '--weights', str(WEIGHTS), '--stage', stage, *map(str, options)]
    print('Running stage', repr(stage), 'in the isolated environment; log:', log, flush=True)
    with log.open('a', encoding='utf-8') as output:
        process = subprocess.Popen(command, env=ENV, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 4000:
                print(line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        detail = 'see the log above'
        if error_file.is_file():
            error = json.loads(error_file.read_text(encoding='utf-8'))
            detail = error['type'] + ': ' + error['message']
        raise RuntimeError(f'Stage {{stage!r}} failed (exit {{process.returncode}}): {{detail}}')


def load_record(name):
    \"\"\"A JSON record a stage wrote to ROOT/outputs.\"\"\"
    return json.loads((ROOT / 'outputs' / name).read_text(encoding='utf-8'))


def show_image(name, caption=None):
    \"\"\"Display an image a stage wrote to ROOT/outputs.\"\"\"
    path = ROOT / 'outputs' / name
    print(caption or name, '->', path)
    display(Image(filename=str(path)))


def show_video(name):
    \"\"\"Play an mp4 a stage wrote to ROOT/outputs (embedded in this session's output; a few MB).\"\"\"
    path = ROOT / 'outputs' / name
    print(name, '->', path)
    display(Video(filename=str(path), embed=True, width=768))"""

WEIGHTS_CELL = """# @title Infrastructure: stage and digest-verify the two pinned snapshots (about {total_gb:.1f} GB)
run_stage('weights')"""


def _probe(modules: list[str]) -> str:
    versions = ", ".join(f"'{m}': {m}.__version__" for m in modules)
    return f"import json, platform, {', '.join(modules)}; print(json.dumps({{'python': platform.python_version(), {versions}, 'torch_cuda': torch.version.cuda, 'cuda': torch.cuda.is_available()}}))"


def _infrastructure_code(key: str, ctx: dict[str, Any], template: dict[str, Any]) -> str:
    gate = template["runtime_gate"]
    if key == "check":
        return CHECK_CELL.format(
            stem=template["stem"], min_gpu_mib=int(gate["min_gpu_mib"]), min_cc=float(gate["min_compute_capability"]),
            min_ram_gb=float(gate["min_available_ram_gb"]), peak_ram_gb=gate["measured_peak_ram_gb"],
            weights_gb=round(ctx["total_gb"] + 0.5, 1), env_gb=float(gate["environment_gb"]),
        )
    if key == "carrier":
        return CARRIER_CELL.format(files=repr(ctx["files"]), hashes=repr(ctx["hashes"]), source=SOURCE_RECORD)
    if key == "install":
        uv = template["uv"]
        return INSTALL_CELL.format(
            uv_url=uv["url"], uv_bytes=int(uv["bytes"]), uv_sha256=uv["sha256"], uv_version=uv["version"],
            python=template["managed_python"], lock=template["lock"], torch_index=template["torch_index"],
            split_source=inspect.getsource(split_lock).rstrip(), phase_a=list(PHASE_A_PACKAGES), probe=_probe(template["runtime_modules"]),
            n_locked=ctx["lock_packages"], runner=template["stage_runner"],
        )
    if key == "weights":
        return WEIGHTS_CELL.format(total_gb=ctx["total_gb"])
    raise SystemExit(f"unknown setup cell {key!r}")


def render(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    ctx = load_context(repo, template, revision)
    for field, allowed in (("profile", PROFILES), ("mode", MODES), ("kind", KINDS)):
        if template[field] not in allowed:
            raise SystemExit(f"template {field} {template[field]!r} is not one of {allowed}")
    stem = template["stem"]
    fmt = {
        "stem": stem, "n_locked": ctx["lock_packages"], "total_gb": f"{ctx['total_gb']:.1f}",
        **{k: ctx[k] for k in IDENTITY_NAMES},
    }
    cells: list[dict[str, Any]] = []

    def add(cell: dict[str, Any]) -> None:
        cell["id"] = f"{stem}-{len(cells):02d}"
        cells.append(cell)

    badges = " ".join(f"[![{alt}]({img})]({link})" for alt, img, link in template["badges"])
    header = (
        f"# {template['title']}\n\n{badges}\n\n"
        f"{template['notice'].strip()}\n\n"
        f"**Kind:** `{template['kind']}`  \n"
        f"**Profile:** `{template['profile']}`  \n"
        f"**Mode:** `{template['mode']}`  \n"
        f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)  \n"
        f"**Capability:** {template['capability']}\n\n"
        f"**This notebook is standalone.** Section 2 carries the repository's package ({len(ctx['modules'])} modules under "
        f"`src/{ctx['pkg']}/`, at revision `{ctx['revision'][:12]}`), the stage runner, the hash-locked requirements "
        f"({ctx['lock_packages']} packages), the two pinned snapshot manifests and the licence, and verifies every carried file "
        f"against its SHA-256 before use. Nothing is installed into the notebook kernel: a pinned `uv` (checked by size and "
        f"SHA-256) builds an isolated Python environment from the lock with `--require-hashes`, and every stage runs there in "
        f"its own process, so the hosted runtime's own packages are never replaced and no restart is needed. Its only external "
        f"dependencies are PyPI, the managed CPython build that `uv` downloads, the Hugging Face Hub at the two immutable "
        f"revisions named below (~{ctx['total_gb']:.1f} GB, digest-verified before loading), and one pinned Wikimedia Commons "
        f"photograph. It was generated by `tools/build_notebook.py` ({GENERATOR_VERSION}); edit the repository and regenerate "
        f"rather than editing cells.\n\n"
        f"**Run all:** {template['run_all'].strip().format(**fmt)}\n\n"
        f"**Bring Your Own Data:** {template['byod'].strip().format(**fmt)}\n\n"
        f"{template['intro'].strip().format(**fmt)}\n\n"
        f"**Learning objectives:** {template['learning_objectives'].strip()}\n\n"
        f"**This notebook does not demonstrate:** {template['exclusions'].strip()}"
    )
    add(_md(header))
    for opening in template["guided"]["opening"]:
        add(_md(opening.format(**fmt)))

    prereq = list(template["prerequisites"]) + [
        f"- **External access:** the Hugging Face Hub, to fetch `{ctx['MODEL_ID']}` at revision `{ctx['MODEL_REVISION'][:12]}…` and "
        f"`{ctx['I2V_ID']}` at revision `{ctx['I2V_REVISION'][:12]}…` (~{ctx['total_gb']:.1f} GB in total, every file named and "
        f"digested in the carried manifests); PyPI (`files.pythonhosted.org`), for the pinned `uv` wheel and the "
        f"{ctx['lock_packages']} hash-locked packages; the managed CPython build (python-build-standalone) that `uv` downloads; "
        "and `upload.wikimedia.org`, for the pinned sample photograph. No repository clone and no credentials are required; "
        "nothing is installed from this repository."
    ]
    add(_md("## Prerequisites\n\n" + "\n".join(p.format(**fmt) for p in prereq)))

    for setup in template["setup"]:
        add(_md(setup["md"].format(**fmt)))
        metadata: dict[str, Any] = dict(INFRASTRUCTURE_METADATA)
        if setup["cell"] == "carrier":
            metadata["dimer"] = {"embedded_sources": True, "files": dict(ctx["hashes"])}
        add(_code(_infrastructure_code(setup["cell"], ctx, template), metadata))
        if setup.get("after"):
            add(_md(setup["after"].format(**fmt)))

    for stage in template["cells"]:
        add(_md(stage["md"].format(**fmt)))
        if stage.get("code"):
            add(_code(stage["code"].format(**fmt)))
        if stage.get("after"):
            add(_md(stage["after"].format(**fmt)))
    for closing in template["closing"]:
        add(_md(closing.format(**fmt)))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "name": template["notebook_name"], "provenance": []},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
            "dimer": {
                "notebook_kind": template["kind"],
                "notebook_profile": template["profile"],
                "notebook_mode": template["mode"],
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                "requires_dimer_worker": False,
                "supported_runtimes": ["Kaggle T4 GPU (Linux x86_64, >= 24 GB available host RAM)"],
                "environment": "isolated hash-locked uv environment; nothing installed into the kernel",
                "models": [
                    {"id": ctx["MODEL_ID"], "revision": ctx["MODEL_REVISION"], "capability": "text-to-video"},
                    {"id": ctx["I2V_ID"], "revision": ctx["I2V_REVISION"], "capability": "image-to-video"},
                ],
                "generated_from": {
                    "repository": template["repo_name"],
                    "revision": ctx["revision"],
                    "module": f"src/{ctx['pkg']}/pipeline.py",
                    "modules": ctx["modules"],
                    "module_sha256": ctx["module_sha256"],
                    "files": dict(ctx["hashes"]),
                    "generator": GENERATOR_VERSION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, default=None, help="default: <repo>/tools/notebook_template.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/tutorials/<notebook_name>")
    parser.add_argument("--check", action="store_true", help="exit 1 if the existing notebook differs from the generator output")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    template = load_template(args.template or repo / "tools" / "notebook_template.py")
    out = args.out or repo / "tutorials" / template["notebook_name"]
    if args.check:
        # The recorded revision is a provenance label carried through the check; drift is caught by content — a changed
        # carried file changes the carrier cell and its hashes, so the byte comparison fails regardless of the label.
        rendered = to_bytes(render(repo, template, recorded_revision(out)))
        current = out.read_bytes().replace(b"\r\n", b"\n") if out.exists() else b""
        if current != rendered:
            print(f"STALE: {out} differs from the generator output; run tools/build_notebook.py", file=sys.stderr)
            return 1
        print(f"OK: {out} is up to date ({len(rendered)} bytes)")
        return 0
    rendered = to_bytes(render(repo, template))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rendered)
    print(f"wrote {out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
