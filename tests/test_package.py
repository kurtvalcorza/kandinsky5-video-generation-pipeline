"""Unit tests for the carried package: input contract, frame checks, manifests and the import boundary."""
# ruff: noqa: E501

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import kandinsky5_video_pipeline as k5
from conftest import MODEL_LIBRARIES, ROOT, moving_frames, synthetic_image
from kandinsky5_video_pipeline import checks, inputs, pipeline

PACKAGE = ROOT / "src" / "kandinsky5_video_pipeline"


# ---- import boundary -------------------------------------------------------------------------------------------------


def test_no_module_level_model_imports() -> None:
    for path in PACKAGE.glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_inputs_are_rejected_before_model_imports(forbid_model_imports) -> None:
    with pytest.raises(ValueError, match="shorter than"):
        k5.validate_prompt("  ")
    with pytest.raises(ValueError, match="4k\\+1"):
        k5.validate_settings(height=512, width=768, num_frames=48, steps=50, guidance_scale=5.0, seed=42)
    with pytest.raises(ValueError, match="shorter side"):
        k5.validate_image(Image.new("RGB", (200, 150)))


# ---- input contract -----------------------------------------------------------------------------------------------------


def test_defaults_are_valid_and_inside_the_measured_envelope() -> None:
    settings = k5.validate_settings(height=k5.HEIGHT, width=k5.WIDTH, num_frames=k5.NUM_FRAMES, steps=k5.STEPS, guidance_scale=k5.GUIDANCE_SCALE, seed=k5.SEED)
    assert settings == {"height": 512, "width": 768, "num_frames": 49, "steps": 50, "guidance_scale": 5.0, "seed": 42, "fps": 24}
    for prompt in (k5.T2V_PROMPT, k5.I2V_PROMPT, k5.NEGATIVE_PROMPT):
        assert k5.validate_prompt(prompt) == " ".join(prompt.split())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"height": 520}, "multiple of 16"),
        ({"height": 768, "width": 768}, "exceeds"),
        ({"num_frames": 53}, "at most 49"),
        ({"steps": 0}, "1..50"),
        ({"guidance_scale": 12.0}, "1.0..10.0"),
        ({"seed": -1}, "2\\*\\*32"),
    ],
)
def test_settings_refusals_name_the_rule(kwargs: dict, message: str) -> None:
    base = {"height": 512, "width": 768, "num_frames": 49, "steps": 50, "guidance_scale": 5.0, "seed": 42}
    with pytest.raises(ValueError, match=message):
        k5.validate_settings(**{**base, **kwargs})


def test_prompt_refusals() -> None:
    with pytest.raises(TypeError):
        k5.validate_prompt(None)
    with pytest.raises(ValueError, match="limit is 1000"):
        k5.validate_prompt("a" * 1001)
    with pytest.raises(ValueError, match="control characters"):
        k5.validate_prompt("abc\x07def")


def test_image_refusals_and_crop_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="width/height"):
        k5.validate_image(Image.new("RGB", (2000, 400)))
    with pytest.raises(ValueError, match="longer side"):
        k5.validate_image(Image.new("RGB", (9000, 5000)))
    # The pinned sample is 3264 x 1840: wider than 3:2, so the crop removes the sides.
    prepared, report = k5.prepare_image(Image.new("RGB", (3264, 1840)))
    assert prepared.size == (768, 512)
    assert report["resized_to"] == [908, 512] and report["crop_box"] == [70, 0, 838, 512]
    assert 0.84 < report["fraction_of_resized_area_kept"] < 0.85
    path = tmp_path / "x.png"
    synthetic_image().save(path)
    image, facts = k5.load_image_file(path)
    assert facts["format"] == "PNG" and image.mode == "RGB"
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        k5.load_image_file(tmp_path / "bad.png")
    with pytest.raises(ValueError, match="not found"):
        k5.load_image_file(tmp_path / "missing.jpg")


# ---- frame checks -------------------------------------------------------------------------------------------------------


def test_frame_checks_pass_on_a_moving_clip() -> None:
    report = k5.frame_checks(moving_frames(), expected_frames=5, height=256, width=256)
    assert report["passed"], report["checks"]


def test_frame_checks_catch_each_failure_mode() -> None:
    clip = moving_frames()
    blank = clip.copy()
    blank[2] = 0.5
    assert not k5.frame_checks(blank, expected_frames=5, height=256, width=256)["checks"]["no_blank_frames"]
    nonfinite = clip.copy()
    nonfinite[1, 0, 0, 0] = np.nan
    assert not k5.frame_checks(nonfinite, expected_frames=5, height=256, width=256)["checks"]["all_values_finite"]
    static = np.repeat(clip[:1], 5, axis=0)
    assert not k5.frame_checks(static, expected_frames=5, height=256, width=256)["checks"]["clip_has_motion"]
    cut = clip.copy()
    cut[3:] = 1.0 - cut[3:]
    assert not k5.frame_checks(cut, expected_frames=5, height=256, width=256)["checks"]["no_abrupt_cuts"]
    assert not k5.frame_checks(clip, expected_frames=49, height=256, width=256)["checks"]["shape_as_requested"]


def test_first_frame_fidelity_and_alignment() -> None:
    clip = checks.to_uint8(moving_frames())
    assert k5.first_frame_fidelity(clip[0], clip[0])["psnr_db"] == float("inf")
    assert k5.first_frame_fidelity(clip[0], clip[4])["mean_abs_diff"] > 0
    summary = k5.alignment_summary([0.30, 0.31, 0.29], [0.20, 0.33, 0.18])
    assert summary["margin"] == pytest.approx(0.0633, abs=1e-4)
    assert summary["frames_closer_to_own_prompt"] == pytest.approx(0.6667, abs=1e-4)
    with pytest.raises(ValueError):
        k5.alignment_summary([], [])


# ---- manifests and pins ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "model_id", "revision"), [(pipeline.MODEL_KEY, pipeline.MODEL_ID, pipeline.MODEL_REVISION), (pipeline.I2V_KEY, pipeline.I2V_ID, pipeline.I2V_REVISION)])
def test_manifests_match_the_package_identity(key: str, model_id: str, revision: str) -> None:
    manifest = pipeline.read_manifest(ROOT / "weights" / key / pipeline.MANIFEST_NAME)
    assert (manifest["modelId"], manifest["revision"]) == (model_id, revision)
    assert all(f["path"].endswith(pipeline.SAFE_SUFFIXES) for f in manifest["files"])


def test_t2v_manifest_holds_every_component_and_i2v_only_its_own() -> None:
    t2v = pipeline.read_manifest(ROOT / "weights" / pipeline.MODEL_KEY / pipeline.MANIFEST_NAME)
    i2v = pipeline.read_manifest(ROOT / "weights" / pipeline.I2V_KEY / pipeline.MANIFEST_NAME)
    t2v_paths = {f["path"] for f in t2v["files"]}
    for component in ("text_encoder/", "text_encoder_2/model.safetensors", "tokenizer/", "tokenizer_2/", "vae/", "transformer/", "scheduler/"):
        assert any(p.startswith(component) for p in t2v_paths), component
    assert len([p for p in t2v_paths if p.startswith("text_encoder/model-")]) == 5
    assert {f["path"] for f in i2v["files"]} == {"model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"}
    shared = set(i2v["sharedComponentsFrom"]["paths"])
    assert shared == t2v_paths - {"model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"}
    transformers = {m["modelKey"]: next(f["sha256"] for f in m["files"] if f["path"] == "transformer/diffusion_pytorch_model.safetensors") for m in (t2v, i2v)}
    assert transformers[pipeline.MODEL_KEY].startswith("9bd1cb1e") and transformers[pipeline.I2V_KEY].startswith("0bfa417e")
    assert t2v["totalBytes"] + i2v["totalBytes"] == 28_442_425_630


def test_manifest_refuses_pickle_formats_and_traversal(tmp_path: Path) -> None:
    good = json.loads((ROOT / "weights" / pipeline.I2V_KEY / pipeline.MANIFEST_NAME).read_text(encoding="utf-8"))
    for bad_path, message in (("text_encoder_2/pytorch_model.bin", "pickle-capable"), ("../escape.json", "unsafe path")):
        manifest = {**good, "files": [*good["files"], {"path": bad_path, "bytes": 1, "sha256": "0" * 64}], "totalBytes": good["totalBytes"] + 1}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            pipeline.read_manifest(path)


def test_verify_snapshot_detects_a_changed_file(tmp_path: Path) -> None:
    data = b'{"a": 1}\n'
    import hashlib

    manifest = {"format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": "k", "modelId": "o/m", "revision": "r", "files": [{"path": "c.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}], "totalBytes": len(data)}
    (tmp_path / pipeline.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert pipeline.missing_files(tmp_path) == ["c.json"]
    (tmp_path / "c.json").write_bytes(data)
    assert pipeline.verify_snapshot(tmp_path, "o/m", "r")["revision"] == "r"
    (tmp_path / "c.json").write_bytes(b'{"a": 2}\n')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pipeline.verify_snapshot(tmp_path, "o/m", "r")
    with pytest.raises(ValueError, match="expected"):
        pipeline.verify_snapshot(tmp_path, "o/other", "r")


def test_sample_image_pin_is_complete() -> None:
    assert pipeline.SAMPLE_IMAGE_URL.startswith("https://upload.wikimedia.org/")
    assert len(pipeline.SAMPLE_IMAGE_SHA256) == 64 and pipeline.SAMPLE_IMAGE_BYTES == 2182680
    assert "CC0" in pipeline.SAMPLE_IMAGE_LICENSE
    assert inputs.INPUT_SCHEMA["validation"]["image"].startswith("JPEG/PNG/WEBP")
