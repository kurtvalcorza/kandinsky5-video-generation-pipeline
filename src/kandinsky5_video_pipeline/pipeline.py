"""Model identity, pinned snapshots and the pinned sample image for the Kandinsky 5.0 Video Lite tutorial.

Nothing here imports a model library at module level (import boundary): torch, diffusers, transformers and
huggingface_hub are imported inside the functions that need them, so validation and refusals run without them.

Two Hub snapshots make one download set. The text-to-video repository supplies every shared component (the
Qwen2.5-VL text encoder, the CLIP text encoder, both tokenizers and the HunyuanVideo VAE) plus its own transformer;
the image-to-video repository contributes only its transformer, scheduler config and pipeline index, because its
shared components are byte-identical to the text-to-video ones at the pinned revisions (recorded in its manifest).
"""
# ruff: noqa: E501  -- contract strings and refusal messages are kept on one line so they read as one rule

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Primary model (text-to-video, supervised fine-tuned "sft" Lite checkpoint, 5-second variant).
MODEL_ID = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
MODEL_REVISION = "5c74c11b3ecfb96249b588d17aebd59f3fd9a080"
MODEL_LICENSE = "MIT (Kandinsky 5.0 card metadata); bundled HunyuanVideo VAE under the Tencent Hunyuan Community License"
MODEL_KEY = "kandinsky5-t2v-lite-sft-5s"

# Second capability (image-to-video Lite checkpoint, 5-second variant).
I2V_ID = "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers"
I2V_REVISION = "8d965626ab76734922b00e359ca2d8faab2e0417"
I2V_LICENSE = MODEL_LICENSE
I2V_KEY = "kandinsky5-i2v-lite-5s"

MANIFEST_NAME = "dimer-base-manifest.json"
MANIFEST_FORMAT = "dimer_hf_snapshot"
SAFE_SUFFIXES = (".safetensors", ".json", ".txt")
REFUSED_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".msgpack", ".h5")

# Component licences, each checked against its upstream source (see MODEL_CARD.md, "Licences").
COMPONENT_LICENSES = {
    "transformer": "MIT (kandinskylab card metadata; github.com/kandinskylab/kandinsky-5 LICENSE)",
    "vae": "Tencent Hunyuan Community License (HunyuanVideo 3D VAE; territory excludes the EU, the UK and South Korea)",
    "text_encoder": "Apache-2.0 (Qwen2.5-VL-7B-Instruct)",
    "text_encoder_2": "MIT (OpenAI CLIP ViT-L/14; github.com/openai/CLIP LICENSE)",
}

# Default image-to-video input: a CC0 photograph from Wikimedia Commons, pinned by URL, size and SHA-256.
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/5/58/Mayon_Volcano%2C_Albay_%28Toe_%29.jpg"
SAMPLE_IMAGE_PAGE = "https://commons.wikimedia.org/wiki/File:Mayon_Volcano,_Albay_(Toe_).jpg"
SAMPLE_IMAGE_BYTES = 2182680
SAMPLE_IMAGE_SHA256 = "138f5ff946e1e5a228270b838beacaa882983c2fea4ffb7e82bf4eefa1df1c01"
SAMPLE_IMAGE_SHA1 = "7f9bee287aee7e77b56ee8cc28f6a023d814f21d"
SAMPLE_IMAGE_UPLOADED = "2018-05-28T02:08:48Z"
SAMPLE_IMAGE_LICENSE = "CC0 1.0 (Creative Commons Zero, public domain dedication)"
SAMPLE_IMAGE_AUTHOR = "P.A.V. Mahusay (Wikimedia Commons user Pavmahusay), own work"
SAMPLE_IMAGE_NAME = "mayon_volcano_albay_toe.jpg"
USER_AGENT = "kandinsky5-video-generation-pipeline/0.1 (DIMER tutorial; https://github.com/kurtvalcorza/kandinsky5-video-generation-pipeline)"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """SHA-256 of a file, read in 16 MiB chunks so multi-GB shards never sit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    """Load and structurally check one snapshot manifest; refuse unsafe serialization formats (MOD9)."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("formatVersion") != 1:
        raise ValueError(f"{path}: not a {MANIFEST_FORMAT} v1 manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: manifest lists no files")
    seen: set[str] = set()
    for entry in files:
        name = entry["path"]
        if name in seen:
            raise ValueError(f"{path}: duplicate manifest entry {name}")
        seen.add(name)
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"{path}: unsafe path {name}")
        if name.endswith(REFUSED_SUFFIXES):
            raise ValueError(f"{path}: {name} uses a pickle-capable or duplicate format; only safetensors weights are staged")
        if len(entry["sha256"]) != 64 or int(entry["bytes"]) < 0:
            raise ValueError(f"{path}: malformed digest or size for {name}")
    if sum(int(f["bytes"]) for f in files) != manifest["totalBytes"]:
        raise ValueError(f"{path}: totalBytes does not equal the sum of file sizes")
    return manifest


def verify_snapshot(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    """Re-hash every manifest file under ``root``; raise on the first missing file, size or digest mismatch (MOD7, MOD8)."""
    manifest = read_manifest(root / MANIFEST_NAME)
    if (manifest["modelId"], manifest["revision"]) != (model_id, revision):
        raise ValueError(f"{root}: manifest names {manifest['modelId']}@{manifest['revision']}, expected {model_id}@{revision}")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ValueError(f"{root.name}: {entry['path']} is missing; re-run the weights stage")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{root.name}: {entry['path']} has {size} bytes, manifest says {entry['bytes']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{root.name}: {entry['path']} SHA-256 mismatch; delete the file and re-run the weights stage")
    return manifest


def missing_files(root: Path) -> list[str]:
    """Manifest paths absent from ``root`` or of the wrong size (cheap check; the digest check is verify_snapshot)."""
    manifest = read_manifest(root / MANIFEST_NAME)
    return [f["path"] for f in manifest["files"] if not (root / f["path"]).is_file() or (root / f["path"]).stat().st_size != f["bytes"]]


def stage_missing_files(root: Path, *, allow_download: bool = True) -> list[str]:
    """Fetch every absent manifest file from the Hub at the pinned revision (never a branch); return what was fetched."""
    manifest = read_manifest(root / MANIFEST_NAME)
    absent = missing_files(root)
    if absent and not allow_download:
        raise ValueError(f"{root.name}: {len(absent)} files are missing and downloads are disabled")
    if not absent:
        return []
    from huggingface_hub import hf_hub_download

    for name in absent:
        hf_hub_download(manifest["modelId"], name, revision=manifest["revision"], local_dir=str(root))
    return absent


def fetch_sample_image(dest_dir: Path) -> Path:
    """Download the pinned CC0 sample photograph once and refuse it on any size or SHA-256 mismatch (DAT2, MOD8)."""
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / SAMPLE_IMAGE_NAME
    if not (path.is_file() and path.stat().st_size == SAMPLE_IMAGE_BYTES and sha256_file(path) == SAMPLE_IMAGE_SHA256):
        request = urllib.request.Request(SAMPLE_IMAGE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(SAMPLE_IMAGE_BYTES + 1)
        if len(data) != SAMPLE_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != SAMPLE_IMAGE_SHA256:
            raise ValueError(
                f"the sample image at {SAMPLE_IMAGE_URL} no longer matches its pinned size/SHA-256 "
                f"({len(data)} bytes); the notebook will not substitute another image. Use the BYOD branch with your own image."
            )
        path.write_bytes(data)
    return path
