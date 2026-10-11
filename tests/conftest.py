# ruff: noqa: E501
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MODEL_LIBRARIES = {"torch", "torchvision", "diffusers", "transformers", "accelerate", "bitsandbytes", "safetensors", "huggingface_hub"}


def load_tool(name: str):
    """Import a script under tools/ as a module (the tools are not a package)."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_image(width: int = 1024, height: int = 640, seed: int = 0) -> Image.Image:
    """A smooth, seeded RGB image (no model semantics)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([np.sin(x / 40 + seed), np.cos(y / 30 - seed), np.sin((x + y) / 60)], axis=-1)
    array = ((base + rng.normal(0, 0.05, base.shape) + 1.5) / 3.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def moving_frames(frames: int = 5, height: int = 256, width: int = 256, seed: int = 0) -> np.ndarray:
    """Float frames in 0..1 whose pattern shifts a little each frame (finite, not blank, moving, no cuts)."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    clip = [np.stack([np.sin((x + 6 * t) / 20 + seed), np.cos((y + 3 * t) / 25), np.sin((x + y + 9 * t) / 40)], axis=-1) for t in range(frames)]
    return ((np.stack(clip) + 1.0) / 2.0).astype(np.float32)
