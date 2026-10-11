"""CPU pre-flight of the stage runner: every stage after 'weights' runs end to end with the four model factories stubbed.

This exercises validation, the refusal probes, state hand-off between processes' files, frame checks, the evaluation
record, the provenance record and the BYOD tag, on the default path and on a BYOD journey. It is not execution evidence:
no model runs (REL8).
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, load_tool, moving_frames, synthetic_image

ts = load_tool("tutorial_stages")
SMALL = ["--height", "256", "--width", "256", "--frames", "5", "--steps", "2"]


@pytest.fixture
def run_dirs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root, weights = tmp_path / "run", tmp_path / "weights"
    for key in ts.SNAPSHOTS:
        (weights / key).mkdir(parents=True)
        shutil.copyfile(ROOT / "weights" / key / "dimer-base-manifest.json", weights / key / "dimer-base-manifest.json")
    image = tmp_path / "photo.png"
    synthetic_image(1024, 640).save(image)
    calls: list[str] = []

    def encode_prompts(run, prompts):
        calls.append("encode")
        (run.state / run.name(ts.EMBED_CACHE)).write_bytes(b"stub")
        return {"embeddings": {k: {"qwen": [1, 7, 3584], "clip": [1, 768], "finite": True, "clip_tokens": 20, "clip_truncated": False} for k in prompts}, "encoder_load_seconds": 0.0, "gpu_peak_gb": 0.0, "gpu_allocated_after_release_gb": 0.0}

    def generate_clip(run, capability, settings, image_in):
        calls.append(capability)
        assert (capability == "i2v") == (image_in is not None)
        frames = moving_frames(settings["num_frames"], settings["height"], settings["width"], seed=settings["seed"] % 7)
        return frames, {"precision": "stub", "load_seconds": 0.0, "generate_seconds": 0.0, "gpu_peak_gb": 0.0, "first_nonfinite_step": None, "steps_completed": settings["steps"]}

    def clip_similarities(run, clips, prompts, conditioning):
        calls.append("clip")
        result = {cap: {"own": [0.3] * len(clips[cap]), "other": [0.2] * len(clips[cap])} for cap in clips}
        if "i2v" in clips:
            result["i2v_to_conditioning_image"] = [0.9] * len(clips["i2v"])
        return result

    monkeypatch.setattr(ts, "encode_prompts", encode_prompts)
    monkeypatch.setattr(ts, "generate_clip", generate_clip)
    monkeypatch.setattr(ts, "clip_similarities", clip_similarities)
    monkeypatch.setattr(ts, "runtime_versions", lambda: {"python": "stub", "torch": "stub"})
    monkeypatch.setattr(ts, "write_video", lambda frames, path, fps: path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(len(frames))))
    monkeypatch.setattr(ts, "host_rss_peak_gb", lambda: 0.0)
    return root, weights, image


def stage(root: Path, weights: Path, name: str, *extra: str) -> int:
    code = ts.main(["--root", str(root), "--weights", str(weights), "--stage", name, *extra])
    error = root / "state" / f"{name}.error.json"
    assert code == 0, error.read_text(encoding="utf-8") if error.is_file() else f"stage {name} exit {code}"
    return code


def test_default_path_plumbing(run_dirs) -> None:
    root, weights, image = run_dirs
    stage(root, weights, "prepare", *SMALL, "--image", str(image))
    prepared = json.loads((root / "outputs" / "prepare.json").read_text(encoding="utf-8"))
    assert [p["probe"] for p in prepared["probes"]] == ["empty prompt", "image too small", "48 frames (not 4k+1)"]
    assert all("rejected" in p for p in prepared["probes"])
    for name in ("encode", "t2v", "i2v", "evaluate", "report"):
        stage(root, weights, name)
    out = root / "outputs"
    for name in ("kandinsky5_video_t2v.mp4", "kandinsky5_video_i2v.mp4", "kandinsky5_video_t2v_strip.jpg", "i2v_conditioning.png", "evaluation.json", "kandinsky5_video_provenance.json"):
        assert (out / name).is_file(), name
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["capabilities"] == ["t2v", "i2v"]
    assert all(evaluation["frame_checks"][c]["passed"] for c in ("t2v", "i2v"))
    assert evaluation["prompt_alignment"]["t2v"]["margin"] == pytest.approx(0.1)
    assert evaluation["i2v_first_frame_fidelity"]["psnr_db"] > 0
    provenance = json.loads((out / "kandinsky5_video_provenance.json").read_text(encoding="utf-8"))
    assert provenance["models"]["t2v"]["revision"] == "5c74c11b3ecfb96249b588d17aebd59f3fd9a080"
    assert provenance["models"]["i2v"]["revision"] == "8d965626ab76734922b00e359ca2d8faab2e0417"
    assert provenance["settings"]["seed"] == 42 and provenance["journey"] == "default"
    assert len(provenance["outputs"]["t2v"]["sha256"]) == 64
    assert np.load(root / "state" / "i2v_frames.npy").dtype == np.uint8
    assert "token" not in json.dumps(provenance).lower()


def test_blank_frames_stop_the_stage(run_dirs, monkeypatch) -> None:
    root, weights, image = run_dirs
    stage(root, weights, "prepare", *SMALL, "--image", str(image))
    stage(root, weights, "encode")
    monkeypatch.setattr(ts, "generate_clip", lambda run, cap, settings, im: (np.zeros((5, 256, 256, 3), np.float32), {"generate_seconds": 0.0, "gpu_peak_gb": 0.0, "first_nonfinite_step": None}))
    assert ts.main(["--root", str(root), "--weights", str(weights), "--stage", "t2v"]) == 2
    error = json.loads((root / "state" / "t2v.error.json").read_text(encoding="utf-8"))
    assert "blank" in error["message"] and "What did not work" in error["message"]


def test_byod_journey_is_tagged_and_reaches_every_stage(run_dirs) -> None:
    root, weights, image = run_dirs
    options = ["--tag", "byod", "--capabilities", "i2v"]
    stage(root, weights, "prepare", *SMALL, *options, "--image", str(image), "--i2v-prompt", "the waves roll in slowly")
    for name in ("encode", "i2v", "evaluate", "report"):
        stage(root, weights, name, *options)
    out = root / "outputs"
    assert (out / "byod_kandinsky5_video_i2v.mp4").is_file() and not (out / "kandinsky5_video_i2v.mp4").exists()
    provenance = json.loads((out / "byod_kandinsky5_video_provenance.json").read_text(encoding="utf-8"))
    assert provenance["journey"] == "byod" and provenance["capabilities"] == ["i2v"]
    assert provenance["inputs"]["image_source"]["kind"] == "byod" and provenance["inputs"]["sample_image_url"] is None
    assert provenance["inputs"]["prompts"]["i2v"] == "the waves roll in slowly"


def test_byod_refuses_an_incompatible_image_before_any_model(run_dirs, tmp_path: Path) -> None:
    root, weights, _image = run_dirs
    small = tmp_path / "small.png"
    synthetic_image(200, 150).save(small)
    assert ts.main(["--root", str(root), "--weights", str(weights), "--stage", "prepare", "--tag", "byod", "--image", str(small)]) == 2
    error = json.loads((root / "state" / "prepare.error.json").read_text(encoding="utf-8"))
    assert "shorter side must be at least 256" in error["message"]
    assert not (root / "state" / "byod_inputs.json").exists()


def test_a_stage_without_its_inputs_names_the_missing_step(run_dirs) -> None:
    root, weights, _image = run_dirs
    assert ts.main(["--root", str(root), "--weights", str(weights), "--stage", "encode"]) == 2
    error = json.loads((root / "state" / "encode.error.json").read_text(encoding="utf-8"))
    assert "inputs.json is missing" in error["message"]


def test_write_video_passes_pil_frames_so_uint8_is_not_rescaled(monkeypatch, tmp_path: Path) -> None:
    """diffusers 0.40.0 export_to_video computes (frame * 255).astype(uint8) for numpy frames; uint8 input must not take that branch."""
    import sys
    import types

    from PIL import Image

    captured = {}
    fake = types.ModuleType("diffusers.utils")
    fake.export_to_video = lambda frames, path, fps: captured.update(frames=frames, path=path, fps=fps)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake)
    frames = (moving_frames(3, 32, 32) * 255).astype(np.uint8)
    ts.write_video(frames, tmp_path / "x.mp4", 24)
    assert all(isinstance(f, Image.Image) for f in captured["frames"]) and captured["fps"] == 24
    assert np.array_equal(np.asarray(captured["frames"][1]), frames[1])
