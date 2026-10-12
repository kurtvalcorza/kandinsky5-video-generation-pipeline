#!/usr/bin/env python3
"""Lock-coverage audit (NOTEBOOK_SPEC 2.4 ENV13, ENV14) for the isolated environment the notebook builds.

Two checks, both against the committed hash lock (or a copy given with ``--lock``):

1. **Import map.** Every third-party module imported anywhere in the code that runs in the isolated environment — the
   carried package under ``src/`` and the stage runner — must resolve to a distribution pinned in the lock. Imports
   inside function bodies count: they are exactly the lazy imports that fail late on a GPU.
2. **Lazy backends.** Libraries on the inference path import or require further distributions on first use. Each one
   found by reading the pinned library source is listed in ``LAZY_BACKENDS`` with its reason and must also be locked.

With ``--library-root DIR`` (a directory holding the pinned libraries' source, e.g. from
``uv pip install --target DIR --no-deps ...``) the audit also searches the modules on the inference path for
``requires_backends`` / ``is_*_available`` checks and fails on any backend it has not classified.

Usage:
    python tools/audit_lock_coverage.py [--lock PATH] [--library-root DIR]
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "kandinsky5_video_pipeline"
RUNTIME_CODE = [*sorted((ROOT / "src" / PACKAGE).glob("*.py")), ROOT / "tools" / "tutorial_stages.py"]
DEFAULT_LOCK = ROOT / "tutorials" / "requirements-kaggle.lock.txt"

# import name -> distribution name in the lock
IMPORT_TO_DIST = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "PIL": "pillow",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "regex": "regex",
    "tokenizers": "tokenizers",
}

# Distributions required lazily on the inference path, each found by reading the pinned source (diffusers 0.40.0,
# transformers 5.16.1, accelerate 1.14.0, bitsandbytes 0.50.2).
LAZY_BACKENDS = {
    "torchvision": "transformers/models/qwen2_vl/image_processing_qwen2_vl.py and video_processing_qwen2_vl.py import torchvision at module level (Qwen2VLProcessor)",
    "regex": "diffusers/pipelines/kandinsky5/pipeline_kandinsky.py and pipeline_kandinsky_i2v.py `import regex as re`",
    "imageio": "diffusers/utils/export_utils.py export_to_video uses imageio when is_imageio_available()",
    "imageio-ffmpeg": "imageio's ffmpeg plugin needs the imageio-ffmpeg binary to write mp4",
    "accelerate": "transformers/quantizers/quantizer_bnb_4bit.py validate_environment requires is_accelerate_available()",
    "bitsandbytes": "transformers/quantizers/quantizer_bnb_4bit.py validate_environment requires is_bitsandbytes_available()",
    "tokenizers": "CLIPTokenizer / Qwen2 tokenizer fast backends",
    "safetensors": "every weight file is safetensors; the runner saves the prompt-embedding cache with safetensors.torch",
}

# Inference-path modules searched with --library-root, and how each availability check found there is classified.
SEARCHED_MODULES = [
    "diffusers/pipelines/kandinsky5/pipeline_kandinsky.py",
    "diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2v.py",
    "diffusers/models/transformers/transformer_kandinsky.py",
    "diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py",
    "diffusers/utils/export_utils.py",
    "diffusers/video_processor.py",
    "diffusers/image_processor.py",
    "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "transformers/models/qwen2_vl/processing_qwen2_vl.py",
    "transformers/models/qwen2_vl/image_processing_qwen2_vl.py",
    "transformers/models/qwen2_vl/video_processing_qwen2_vl.py",
    "transformers/models/clip/modeling_clip.py",
    "transformers/models/clip/image_processing_clip.py",
    "transformers/models/clip/tokenization_clip.py",
    "transformers/quantizers/quantizer_bnb_4bit.py",
    "transformers/integrations/bitsandbytes.py",
]
CLASSIFIED_BACKENDS = {
    "accelerate": "locked",
    "bitsandbytes": "locked",
    "torch": "locked",
    "imageio": "locked",
    "ftfy": "optional: text clean-up runs only if installed; not needed",
    "torch_xla": "not used: TPU only",
    "opencv": "not used: fallback exporter only when imageio is absent",
    "torch_npu": "not used: Ascend NPU only",
    "torch_hpu": "not used: Gaudi HPU only",
    "torch_xpu": "not used: Intel XPU only",
}


def lock_packages(lock_text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
CUDA_LOCAL_VERSION = "+cu128"


def cuda_stack_problems(lock_text: str) -> list[str]:
    """The lock must hold the CUDA 12.8 stack the Kaggle probe ran: +cu128 torch/torchvision and only -cu12 NVIDIA wheels.

    A CUDA 13 build (PyPI's default torch 2.11.0 Linux wheel, with nvidia-*-cu13 or unsuffixed nvidia-* 13.x wheels) failed
    on Kaggle's CUDA 12.8 host in the fleet's records (libnvJitLink.so.13 not found).
    """
    locked = lock_packages(lock_text)
    problems = []
    for name in ("torch", "torchvision"):
        version = locked.get(name)
        if version is None or not version.endswith(CUDA_LOCAL_VERSION):
            problems.append(f"{name}=={version} is not a {CUDA_LOCAL_VERSION} build from {TORCH_INDEX}")
    for name, version in locked.items():
        if name.startswith("nvidia-") and not name.endswith("-cu12"):
            problems.append(f"{name}=={version} is not a CUDA 12 (-cu12) NVIDIA wheel")
        if "cu13" in name or (name in ("cuda-toolkit", "nvidia-cuda-runtime-cu12") and not version.startswith("12.8")):
            problems.append(f"{name}=={version} is not part of the CUDA 12.8 stack")
    return problems


def install_split_problems(lock_text: str) -> list[str]:
    """V-F2: the notebook's two-phase split covers every lock entry exactly once; phase A is only the +cu128 builds."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    phase_a, phase_b = (lock_packages(text) for text in build.split_lock(lock_text, build.PHASE_A_PACKAGES))
    locked = lock_packages(lock_text)
    problems = []
    if set(phase_a) & set(phase_b) or set(phase_a) | set(phase_b) != set(locked):
        problems.append("the two-phase split does not cover every lock entry exactly once")
    if sorted(phase_a) != sorted(build.PHASE_A_PACKAGES) or not all(v.endswith(CUDA_LOCAL_VERSION) for v in phase_a.values()):
        problems.append(f"phase A (PyTorch cu128 index) must be exactly {build.PHASE_A_PACKAGES} as {CUDA_LOCAL_VERSION} builds, got {phase_a}")
    if any(v.endswith(CUDA_LOCAL_VERSION) for v in phase_b.values()):
        problems.append("phase B (PyPI) contains a +cu128 build that PyPI does not serve")
    return problems


def imported_modules(paths: list[Path]) -> dict[str, list[str]]:
    """`{top-level module: [file:line, ...]}` for every import statement, at any nesting depth."""
    found: dict[str, list[str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                found.setdefault(name.partition(".")[0], []).append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    return found


def audit(lock_path: Path, library_root: Path | None = None) -> list[str]:
    """Return a list of problems (empty when the lock covers everything)."""
    lock_text = lock_path.read_text(encoding="utf-8")
    locked = lock_packages(lock_text)
    problems: list[str] = cuda_stack_problems(lock_text) + install_split_problems(lock_text)
    print(f"CUDA stack: torch {locked.get('torch')}, torchvision {locked.get('torchvision')}, {sum(n.startswith('nvidia-') for n in locked)} nvidia-* wheels")
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    imports = imported_modules(RUNTIME_CODE)
    print(f"import map ({len(RUNTIME_CODE)} files):")
    for module, where in sorted(imports.items()):
        if module in stdlib or module == PACKAGE:
            continue
        dist = IMPORT_TO_DIST.get(module)
        if dist is None:
            problems.append(f"{module} (imported at {where[0]}) has no distribution mapping; add it to IMPORT_TO_DIST and the lock")
            continue
        status = locked.get(dist)
        print(f"  {module:16s} -> {dist:16s} {status or 'MISSING'}  ({len(where)} import sites, first {where[0]})")
        if status is None:
            problems.append(f"{module} -> {dist} is imported by the runtime code but not pinned in {lock_path.name}")
    print("lazy backends:")
    for dist, reason in LAZY_BACKENDS.items():
        status = locked.get(dist)
        print(f"  {dist:16s} {status or 'MISSING'}  ({reason})")
        if status is None:
            problems.append(f"lazy backend {dist} is not pinned in {lock_path.name}: {reason}")
    if library_root is not None:
        print(f"availability checks in {len(SEARCHED_MODULES)} pinned modules under {library_root}:")
        for rel in SEARCHED_MODULES:
            path = library_root / rel
            if not path.is_file():
                problems.append(f"{rel} not found under {library_root}; is it the pinned library version?")
                continue
            text = path.read_text(encoding="utf-8")
            names = set(re.findall(r"is_([a-z0-9_]+?)_available\b", text)) | set(re.findall(r"requires_backends\([^,]+,\s*\[?([^\])]+)", text))
            for name in sorted(names):
                key = name.strip(" '\"")
                verdict = CLASSIFIED_BACKENDS.get(key)
                print(f"  {rel}: {key} -> {verdict or 'UNCLASSIFIED'}")
                if verdict is None:
                    problems.append(f"{rel} checks backend {key!r}, which the audit has not classified")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--library-root", type=Path, default=None)
    args = parser.parse_args(argv)
    problems = audit(args.lock, args.library_root)
    if problems:
        for problem in problems:
            print("FAIL:", problem, file=sys.stderr)
        return 1
    print(f"OK: {args.lock.name} covers every runtime import and lazy backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
