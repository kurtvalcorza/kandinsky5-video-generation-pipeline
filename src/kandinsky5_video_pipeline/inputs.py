"""Input contract and notebook-local validation for both capabilities (VAL1–VAL8, DAT12, DAT19).

Every check here runs before any model library is imported, so a bad prompt, image or setting is refused in seconds
instead of after a 28 GB download or a 20-minute generation. Messages name the failed rule and the fix (UX10).
"""
# ruff: noqa: E501  -- contract strings and refusal messages are kept on one line so they read as one rule

from __future__ import annotations

from pathlib import Path
from typing import Any

# Generation settings measured on a Kaggle T4 by the development probe (see the notebook, "What did not work").
HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 49
STEPS = 50
GUIDANCE_SCALE = 5.0
SEED = 42
FPS = 24

T2V_PROMPT = (
    "A small wooden outrigger fishing boat glides across calm turquoise water at sunrise near a palm-lined beach, "
    "the camera slowly tracking alongside it."
)
I2V_PROMPT = (
    "Low clouds drift slowly across the sky and the palm trees sway in the wind beyond the dark lava field, "
    "while the camera pans slowly to the right."
)
# The negative prompt published with the Kandinsky 5.0 Lite model cards.
NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
)

PROMPT_MIN_CHARS = 3
PROMPT_MAX_CHARS = 1000
IMAGE_MIN_SIDE = 256
IMAGE_MAX_SIDE = 8192
IMAGE_MAX_BYTES = 25 * 1024 * 1024
IMAGE_MIN_ASPECT = 0.5
IMAGE_MAX_ASPECT = 2.5
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
MAX_FRAMES = 49
MAX_PIXELS = HEIGHT * WIDTH
MAX_STEPS = 50

INPUT_SCHEMA: dict[str, Any] = {
    "t2v": {"input": "prompt: str", "output": f"{NUM_FRAMES} RGB frames of {HEIGHT}x{WIDTH} px, an mp4 at {FPS} fps"},
    "i2v": {"input": "prompt: str + image: JPEG/PNG/WEBP", "output": f"{NUM_FRAMES} RGB frames of {HEIGHT}x{WIDTH} px, an mp4 at {FPS} fps"},
    "validation": {
        "prompt": f"{PROMPT_MIN_CHARS}..{PROMPT_MAX_CHARS} characters after trimming, printable text",
        "image": f"{'/'.join(IMAGE_FORMATS)}, at most {IMAGE_MAX_BYTES // 2**20} MiB, shorter side >= {IMAGE_MIN_SIDE} px, longer side <= {IMAGE_MAX_SIDE} px, width/height in {IMAGE_MIN_ASPECT}..{IMAGE_MAX_ASPECT}",
        "settings": f"height and width multiples of 16 with height*width <= {MAX_PIXELS}; num_frames = 4k+1 <= {MAX_FRAMES}; 1 <= steps <= {MAX_STEPS}; 1.0 <= guidance <= 10.0; 0 <= seed < 2**32",
    },
}


def validate_prompt(prompt: Any, *, name: str = "prompt") -> str:
    """Return the trimmed prompt or raise ValueError naming the rule it breaks."""
    if not isinstance(prompt, str):
        raise TypeError(f"{name} must be a string, got {type(prompt).__name__}")
    text = " ".join(prompt.split())
    if len(text) < PROMPT_MIN_CHARS:
        raise ValueError(f"{name} is empty or shorter than {PROMPT_MIN_CHARS} characters; describe the scene and the motion")
    if len(text) > PROMPT_MAX_CHARS:
        raise ValueError(f"{name} has {len(text)} characters; the limit is {PROMPT_MAX_CHARS}. Shorten it: the text encoder keeps at most 512 tokens")
    if any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{name} contains control characters; remove them")
    return text


def validate_settings(*, height: int, width: int, num_frames: int, steps: int, guidance_scale: float, seed: int) -> dict[str, Any]:
    """Check generation settings against the T4 operating envelope (VAL6) before any model loads."""
    problems = []
    for label, value in (("height", height), ("width", width)):
        if not isinstance(value, int) or value < 256 or value % 16:
            problems.append(f"{label}={value!r} must be an integer >= 256 and a multiple of 16")
    if isinstance(height, int) and isinstance(width, int) and height * width > MAX_PIXELS:
        problems.append(f"height*width={height * width} exceeds {MAX_PIXELS} ({HEIGHT}x{WIDTH}), the largest size measured on a T4")
    if not isinstance(num_frames, int) or num_frames < 1 or num_frames > MAX_FRAMES or (num_frames - 1) % 4:
        problems.append(f"num_frames={num_frames!r} must be 4k+1 and at most {MAX_FRAMES} (the VAE compresses time by 4)")
    if not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        problems.append(f"steps={steps!r} must be an integer in 1..{MAX_STEPS}")
    if not isinstance(guidance_scale, int | float) or not 1.0 <= float(guidance_scale) <= 10.0:
        problems.append(f"guidance_scale={guidance_scale!r} must be in 1.0..10.0")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        problems.append(f"seed={seed!r} must be an integer in 0..2**32-1")
    if problems:
        raise ValueError("invalid generation settings: " + "; ".join(problems))
    return {"height": height, "width": width, "num_frames": num_frames, "steps": steps, "guidance_scale": float(guidance_scale), "seed": seed, "fps": FPS}


def validate_image(image: Any, *, name: str = "image") -> dict[str, Any]:
    """Check a decoded PIL image against the image contract; return its facts."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"{name} must be a PIL image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < IMAGE_MIN_SIDE:
        raise ValueError(f"{name} is {width}x{height} px; the shorter side must be at least {IMAGE_MIN_SIDE} px")
    if max(width, height) > IMAGE_MAX_SIDE:
        raise ValueError(f"{name} is {width}x{height} px; the longer side must be at most {IMAGE_MAX_SIDE} px")
    aspect = width / height
    if not IMAGE_MIN_ASPECT <= aspect <= IMAGE_MAX_ASPECT:
        raise ValueError(f"{name} has width/height {aspect:.2f}; use an image between {IMAGE_MIN_ASPECT} and {IMAGE_MAX_ASPECT} so the centre crop keeps the subject")
    return {"width": width, "height": height, "mode": image.mode, "aspect": round(aspect, 3)}


def load_image_file(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Open a user or sample image file with the same checks as an in-memory image (DAT13, DAT19)."""
    from PIL import Image, UnidentifiedImageError

    file = Path(path)
    if not file.is_file():
        raise ValueError(f"image file not found: {file}. Set BYOD_IMAGE_PATH to a file inside this runtime")
    size = file.stat().st_size
    if size > IMAGE_MAX_BYTES:
        raise ValueError(f"{file.name} is {size} bytes; the limit is {IMAGE_MAX_BYTES} bytes ({IMAGE_MAX_BYTES // 2**20} MiB)")
    try:
        with Image.open(file) as handle:
            fmt = handle.format
            image = handle.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{file.name} is not a readable image; use JPEG, PNG or WEBP") from exc
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"{file.name} is {fmt}; use one of {', '.join(IMAGE_FORMATS)}")
    facts = validate_image(image, name=file.name)
    return image, {**facts, "format": fmt, "bytes": size}


def prepare_image(image: Any, *, width: int = WIDTH, height: int = HEIGHT) -> tuple[Any, dict[str, Any]]:
    """Resize to cover ``width x height`` and centre-crop; report exactly what was removed (VAL7)."""
    from PIL import Image

    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(width, round(src_w * scale)), max(height, round(src_h * scale))
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    kept = (width * height) / (new_w * new_h)
    report = {
        "source_size": [src_w, src_h],
        "resized_to": [new_w, new_h],
        "crop_box": [left, top, left + width, top + height],
        "output_size": [width, height],
        "fraction_of_resized_area_kept": round(kept, 4),
        "note": "resized with Lanczos to cover the output size, then centre-cropped; nothing else is changed",
    }
    return cropped, report
