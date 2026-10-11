#!/usr/bin/env python3
"""Index-coverage check for the hash lock (V-F2): can each install phase get every locked wheel from its ONE index?

The notebook installs the lock in two phases, each from a single index with ``--no-deps --require-hashes``:
phase A (``torch``, ``torchvision``) from the PyTorch CUDA 12.8 index, phase B (everything else) from PyPI. This tool
reads only index metadata (no wheel is downloaded) and, for every lock entry, checks that the phase's index serves a
wheel of the locked version that is installable on CPython 3.12 / manylinux_2_28 x86_64 AND whose sha256 is locked.

``--mode mixed`` instead models the earlier single install (``--index-url PyPI --extra-index-url cu128
--index-strategy unsafe-best-match``): any entry for which EITHER index serves an installable wheel of the locked version
whose sha256 is NOT locked is reported as a hazard, because the resolver may pick that file and fail the hash check.
That is what happened to ``triton==3.6.0`` on Kaggle.

Usage:
    python tools/check_index_coverage.py [--lock PATH] [--mode split|mixed]
Needs network access to pypi.org and download.pytorch.org. Not run in CI (network); results are recorded in the
release documentation.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "tutorials" / "requirements-kaggle.lock.txt"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PHASE_A = ("torch", "torchvision")
MAX_GLIBC = (2, 28)
Files = list[tuple[str, str]]  # (filename, sha256)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_entries(lock_text: str) -> dict[str, tuple[str, set[str]]]:
    """`{normalized name: (version, {sha256, ...})}` of every lock entry."""
    entries = {}
    for block in re.split(r"\n(?=[A-Za-z0-9])", lock_text):
        match = re.match(r"([A-Za-z0-9._-]+)==([^\s\\]+)", block)
        if match:
            entries[normalize(match.group(1))] = (match.group(2), set(re.findall(r"--hash=sha256:([0-9a-f]{64})", block)))
    return entries


def wheel_installable(filename: str) -> bool:
    """True for a wheel installable on CPython 3.12, manylinux_2_28 x86_64 (the notebook's isolated environment)."""
    if not filename.endswith(".whl"):
        return False
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return False
    py_tags, abi_tags, plat_tags = (parts[-3].split("."), parts[-2].split("."), parts[-1].split("."))
    python_ok = any(t in ("cp312", "py3", "py312") for t in py_tags) or (
        "abi3" in abi_tags and any(re.fullmatch(r"cp3(\d+)", t) and int(t[3:]) <= 12 for t in py_tags)
    )
    abi_ok = any(t in ("cp312", "abi3", "none") for t in abi_tags)

    def plat_ok(tag: str) -> bool:
        if tag == "any":
            return True
        if not tag.endswith("_x86_64"):
            return False
        legacy = {"manylinux1_x86_64": (2, 5), "manylinux2010_x86_64": (2, 12), "manylinux2014_x86_64": (2, 17)}
        if tag in legacy:
            return legacy[tag] <= MAX_GLIBC
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", tag)
        return bool(match) and (int(match.group(1)), int(match.group(2))) <= MAX_GLIBC

    return python_ok and abi_ok and any(plat_ok(t) for t in plat_tags)


def _get(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "kandinsky5-video-generation-pipeline index-coverage check"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):  # download.pytorch.org answers 403 for a project it does not host
            return None
        raise


def pypi_files(name: str, version: str) -> Files:
    """Files PyPI serves for `name==version`, with their sha256."""
    data = _get(PYPI_JSON.format(name=name, version=urllib.parse.quote(version)))
    if data is None:
        return []
    return [(u["filename"], u["digests"]["sha256"]) for u in json.loads(data)["urls"]]


def torch_index_files(name: str, version: str) -> Files:
    """Files the PyTorch cu128 simple index serves for `name==version`, from the `#sha256=` fragments."""
    data = _get(f"{TORCH_INDEX}/{name}/")
    if data is None:
        return []
    files = []
    for href in re.findall(r'href="([^"]+)"', data.decode("utf-8", "replace")):
        path, _, fragment = href.partition("#sha256=")
        filename = urllib.parse.unquote(path.rsplit("/", 1)[-1])
        parts = filename.split("-")
        if fragment and len(parts) > 2 and parts[1] == version:
            files.append((filename, fragment))
    return files


def check(lock_text: str, mode: str, pypi: Callable[[str, str], Files] = pypi_files, torch_index: Callable[[str, str], Files] = torch_index_files) -> tuple[list[str], list[str]]:
    """Return (report lines, problems). `pypi` and `torch_index` are injectable for offline tests."""
    lines, problems = [], []
    for name, (version, hashes) in sorted(lock_entries(lock_text).items()):
        if mode == "split":
            index_name, fetch = ("cu128", torch_index) if name in PHASE_A else ("pypi", pypi)
            installable = [(f, h) for f, h in fetch(name, version) if wheel_installable(f)]
            served = [f for f, h in installable if h in hashes]
            lines.append(f"{name}=={version} [{index_name}] installable={len(installable)} locked-hash-served={len(served)}")
            if not served:
                problems.append(f"{name}=={version}: index {index_name} serves no installable wheel with a locked sha256")
        else:
            candidates = [("pypi", f, h) for f, h in pypi(name, version) if wheel_installable(f)]
            candidates += [("cu128", f, h) for f, h in torch_index(name, version) if wheel_installable(f)]
            foreign = [(i, f, h) for i, f, h in candidates if h not in hashes]
            lines.append(f"{name}=={version} candidates={len(candidates)} unlocked={len(foreign)}")
            for index_name, filename, digest in foreign:
                problems.append(f"HAZARD {name}=={version}: {index_name} serves installable {filename} sha256:{digest} not in the lock")
            if not any(h in hashes for _, _, h in candidates):
                problems.append(f"{name}=={version}: no index serves a wheel with a locked sha256")
    return lines, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--mode", choices=("split", "mixed"), default="split")
    args = parser.parse_args(argv)
    lines, problems = check(args.lock.read_text(encoding="utf-8"), args.mode)
    for line in lines:
        print(line)
    for problem in problems:
        print("FAIL:", problem, file=sys.stderr)
    print(f"{len(lines)} entries checked in {args.mode} mode; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
