"""Stage runner for the standalone Kandinsky 5.0 Video Lite tutorial (NOTEBOOK_SPEC 2.4 §8.1 isolated environment).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage prepare [--image PATH] [--tag byod]

Nothing is installed into the notebook kernel. Each stage is a separate process, so GPU and host memory are returned
to the system when it exits, and a stage starts from files only: the verified snapshots under ``--weights``, the
prompt-embedding cache written by ``encode``, the frames written by ``t2v``/``i2v`` and the JSON records of earlier
stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a stage
writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises.

Stages: weights -> prepare -> encode -> t2v -> i2v -> evaluate -> report, plus the optional ``activity``.
Model work is isolated in four factory functions (``encode_prompts``, ``generate_clip``, ``clip_similarities`` and
``runtime_versions``) so the CPU tests can replace them with stubs and exercise every stage's plumbing.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "kandinsky5_video"
SNAPSHOTS = ("kandinsky5-t2v-lite-sft-5s", "kandinsky5-i2v-lite-5s")
CAPABILITIES = ("t2v", "i2v")
EMBED_CACHE = "prompt_embeds.safetensors"
ACTIVITY_SEED = 7


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, snapshots under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.tag = options.tag
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def name(self, base: str) -> str:
        """A per-journey file name: the default path uses ``base``; BYOD prefixes its tag so nothing is overwritten."""
        return f"{self.tag}_{base}" if self.tag else base

    def snapshot(self, key: str) -> Path:
        return self.weights / key

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / self.name(name)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / self.name(name)
        if not path.is_file():
            raise RuntimeError(f"{path.name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / self.name(name)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def host_rss_peak_gb() -> float | None:
    """Peak resident memory of this stage process (Linux ru_maxrss is KiB); None where ``resource`` does not exist."""
    try:
        import resource
    except ImportError:  # Windows, where only the CPU tests run
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 2)


def gpu_peak_gb() -> float | None:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def require_snapshot_files(run: Run, key: str) -> Path:
    """Cheap presence-and-size check before loading; the full SHA-256 check ran in the 'weights' stage."""
    from kandinsky5_video_pipeline import missing_files

    root = run.snapshot(key)
    absent = missing_files(root)
    if absent:
        raise RuntimeError(f"{key}: {len(absent)} files are missing or changed size since the 'weights' stage (first: {absent[0]}); re-run Section 3")
    return root


# --------------------------------------------------------------------------------------------------
# model factories (the CPU tests replace these with stubs)
# --------------------------------------------------------------------------------------------------


def runtime_versions() -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import diffusers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def encode_prompts(run: Run, prompts: dict[str, str]) -> dict[str, Any]:
    """Load Qwen2.5-VL in 4-bit NF4 and CLIP in float16, encode every prompt once, save the embeddings, free both."""
    import gc

    import safetensors.torch
    import torch
    from diffusers import Kandinsky5T2VPipeline
    from transformers import (
        BitsAndBytesConfig,
        CLIPTextModel,
        CLIPTokenizer,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLProcessor,
    )

    root = require_snapshot_files(run, SNAPSHOTS[0])
    t0 = time.perf_counter()
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(root, subfolder="text_encoder", quantization_config=quantization, torch_dtype=torch.float16, device_map={"": 0})
    tokenizer = Qwen2VLProcessor.from_pretrained(root, subfolder="tokenizer")
    text_encoder_2 = CLIPTextModel.from_pretrained(root, subfolder="text_encoder_2", torch_dtype=torch.float16).to("cuda")
    tokenizer_2 = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer_2")
    load_seconds = round(time.perf_counter() - t0, 1)
    encoder = Kandinsky5T2VPipeline(transformer=None, vae=None, text_encoder=text_encoder, tokenizer=tokenizer, text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, scheduler=None)
    tensors: dict[str, Any] = {}
    shapes: dict[str, Any] = {}
    with torch.inference_mode():
        for key, prompt in prompts.items():
            qwen, clip, cu_seqlens = encoder.encode_prompt(prompt, device=torch.device("cuda"), dtype=torch.float16)
            tensors[f"{key}.qwen"] = qwen.detach().cpu().contiguous()
            tensors[f"{key}.clip"] = clip.detach().cpu().contiguous()
            tensors[f"{key}.cu_seqlens"] = cu_seqlens.detach().cpu().contiguous()
            finite = bool(torch.isfinite(qwen).all() and torch.isfinite(clip).all())
            clip_tokens = len(tokenizer_2(prompt)["input_ids"])
            shapes[key] = {"qwen": list(qwen.shape), "clip": list(clip.shape), "finite": finite, "clip_tokens": clip_tokens, "clip_truncated": clip_tokens > tokenizer_2.model_max_length}
            if not finite:
                raise RuntimeError(f"the {key} prompt embedding is not finite; see Troubleshooting (encoder precision)")
    safetensors.torch.save_file(tensors, run.state / run.name(EMBED_CACHE))
    peak = gpu_peak_gb()
    del encoder, text_encoder, text_encoder_2
    gc.collect()
    torch.cuda.empty_cache()
    return {"embeddings": shapes, "encoder_load_seconds": load_seconds, "gpu_peak_gb": peak, "gpu_allocated_after_release_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


def generate_clip(run: Run, capability: str, settings: dict[str, Any], image: Any) -> tuple[Any, dict[str, Any]]:
    """Load the capability's transformer and the shared VAE in float16, generate one clip from the cached embeddings."""
    import gc

    import safetensors.torch
    import torch
    from diffusers import (
        AutoencoderKLHunyuanVideo,
        FlowMatchEulerDiscreteScheduler,
        Kandinsky5I2VPipeline,
        Kandinsky5T2VPipeline,
        Kandinsky5Transformer3DModel,
    )

    shared = require_snapshot_files(run, SNAPSHOTS[0])
    own = shared if capability == "t2v" else require_snapshot_files(run, SNAPSHOTS[1])
    cache = safetensors.torch.load_file(run.state / run.name(EMBED_CACHE))

    def embedding(key: str) -> tuple[Any, Any, Any]:
        return cache[f"{key}.qwen"].to("cuda", torch.float16), cache[f"{key}.clip"].to("cuda", torch.float16), cache[f"{key}.cu_seqlens"].to("cuda")

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vae = AutoencoderKLHunyuanVideo.from_pretrained(shared, subfolder="vae", torch_dtype=torch.float16)
    vae.enable_tiling()  # decodes the 49-frame latent in tiles so the decoder fits beside the transformer
    transformer = Kandinsky5Transformer3DModel.from_pretrained(own, subfolder="transformer", torch_dtype=torch.float16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(own, subfolder="scheduler")
    pipeline_class = Kandinsky5T2VPipeline if capability == "t2v" else Kandinsky5I2VPipeline
    pipe = pipeline_class(transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, scheduler=scheduler).to("cuda")
    load_seconds = round(time.perf_counter() - t0, 1)
    steps_log: list[dict[str, Any]] = []

    def track(_pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = kwargs["latents"]
        steps_log.append({"step": int(step), "t": round(float(timestep), 2), "finite_fraction": round(float(torch.isfinite(latents).float().mean()), 6)})
        return kwargs

    q, c, cu = embedding(capability)
    nq, nc, ncu = embedding("negative")
    extra = {"image": image} if capability == "i2v" else {}
    t1 = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(
            prompt=None, prompt_embeds_qwen=q, prompt_embeds_clip=c, prompt_cu_seqlens=cu,
            negative_prompt_embeds_qwen=nq, negative_prompt_embeds_clip=nc, negative_prompt_cu_seqlens=ncu,
            height=settings["height"], width=settings["width"], num_frames=settings["num_frames"],
            num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
            generator=torch.Generator("cuda").manual_seed(settings["seed"]), output_type="np",
            callback_on_step_end=track, callback_on_step_end_tensor_inputs=["latents"], **extra,
        ).frames[0]
    generate_seconds = round(time.perf_counter() - t1, 1)
    info = {
        "precision": "float16 (transformer and VAE)",
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "gpu_peak_gb": gpu_peak_gb(),
        "first_nonfinite_step": next((s for s in steps_log if s["finite_fraction"] < 1.0), None),
        "steps_completed": len(steps_log),
    }
    del pipe, transformer, vae
    gc.collect()
    torch.cuda.empty_cache()
    return frames, info


def clip_similarities(run: Run, clips: dict[str, Any], prompts: dict[str, str], conditioning: Any) -> dict[str, Any]:
    """CLIP ViT-L/14 cosine similarity of every frame to each prompt, and of the image-to-video frames to the conditioning image."""
    import gc

    import torch
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

    root = require_snapshot_files(run, SNAPSHOTS[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(root, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(root, subfolder="text_encoder_2")
    tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer_2")

    def unit(features: Any) -> Any:
        # transformers 5.x returns a BaseModelOutputWithPooling whose pooler_output is the projected embedding.
        tensor = features.pooler_output if hasattr(features, "pooler_output") else features
        return torch.nn.functional.normalize(tensor.float(), dim=-1)

    def embed_images(images: list[Any]) -> Any:
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device, dtype)
        return unit(model.get_image_features(pixel_values=pixels))

    with torch.inference_mode():
        tokens = tokenizer([prompts[k] for k in CAPABILITIES], padding=True, truncation=True, return_tensors="pt").to(device)
        text = unit(model.get_text_features(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]))
        result: dict[str, Any] = {}
        for i, cap in enumerate(CAPABILITIES):
            if cap not in clips:
                continue
            frames = embed_images(list(clips[cap]))
            sims = (frames @ text.T).cpu().numpy()
            result[cap] = {"own": sims[:, i].tolist(), "other": sims[:, 1 - i].tolist()}
        if "i2v" in clips:
            reference = embed_images([conditioning])
            result["i2v_to_conditioning_image"] = (embed_images(list(clips["i2v"])) @ reference.T).cpu().numpy().ravel().tolist()
    model = None  # drop the last reference so the scorer's GPU memory can be freed before the stage exits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifests, fetch absent files at the pinned revisions, re-hash every file."""
    from kandinsky5_video_pipeline import (
        I2V_ID,
        I2V_LICENSE,
        I2V_REVISION,
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        read_manifest,
        stage_missing_files,
        verify_snapshot,
    )

    plan = ((SNAPSHOTS[0], MODEL_ID, MODEL_REVISION, MODEL_LICENSE), (SNAPSHOTS[1], I2V_ID, I2V_REVISION, I2V_LICENSE))
    records = []
    for key, model_id, revision, license_name in plan:
        carried = run.root / "weights" / key / MANIFEST_NAME
        manifest = read_manifest(carried)
        if (manifest["modelId"], manifest["revision"]) != (model_id, revision):
            raise RuntimeError(f"carried {key} manifest does not name the identity pinned by the package; regenerate the notebook")
        target = run.snapshot(key)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(carried, target / MANIFEST_NAME)
        print({"model_id": model_id, "revision": revision, "files": len(manifest["files"]), "total_gb": round(manifest["totalBytes"] / 1e9, 2)}, flush=True)
        t0 = time.perf_counter()
        fetched = stage_missing_files(target, allow_download=True)
        download_seconds = round(time.perf_counter() - t0, 1)
        t0 = time.perf_counter()
        verify_snapshot(target, model_id, revision)
        verify_seconds = round(time.perf_counter() - t0, 1)
        print({"snapshot": key, "fetched_files": len(fetched), "download_seconds": download_seconds, "verified_files": len(manifest["files"]), "verify_seconds": verify_seconds}, flush=True)
        records.append({"key": key, "id": model_id, "revision": revision, "license": license_name, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"], "fetched": len(fetched), "manifest": manifest.get("sharedComponentsFrom")})
    run.write_output("weights.json", {"snapshots": records})


def stage_prepare(run: Run) -> None:
    """Section 4: obtain and validate both inputs and the settings; show three refusals; write the prepared image."""
    from PIL import Image

    from kandinsky5_video_pipeline import (
        INPUT_SCHEMA,
        NEGATIVE_PROMPT,
        SAMPLE_IMAGE_AUTHOR,
        SAMPLE_IMAGE_LICENSE,
        SAMPLE_IMAGE_PAGE,
        SAMPLE_IMAGE_SHA256,
        fetch_sample_image,
        load_image_file,
        prepare_image,
        sha256_file,
        validate_image,
        validate_prompt,
        validate_settings,
    )

    opts = run.options
    settings = validate_settings(height=opts.height, width=opts.width, num_frames=opts.frames, steps=opts.steps, guidance_scale=opts.guidance, seed=opts.seed)
    prompts = {"t2v": validate_prompt(opts.t2v_prompt, name="T2V prompt"), "i2v": validate_prompt(opts.i2v_prompt, name="I2V prompt"), "negative": validate_prompt(NEGATIVE_PROMPT, name="negative prompt")}
    if opts.image:
        image_path = Path(opts.image).resolve()
        source = {"kind": "byod", "file": image_path.name, "sha256": sha256_file(image_path)}
    else:
        image_path = fetch_sample_image(run.weights / "sample")
        source = {"kind": "sample", "url_page": SAMPLE_IMAGE_PAGE, "license": SAMPLE_IMAGE_LICENSE, "author": SAMPLE_IMAGE_AUTHOR, "sha256": SAMPLE_IMAGE_SHA256}
    image, facts = load_image_file(image_path)
    prepared, crop = prepare_image(image, width=settings["width"], height=settings["height"])
    prepared_path = run.out / run.name("i2v_conditioning.png")
    prepared.save(prepared_path)
    print({"settings": settings})
    print({"prompts": {k: v for k, v in prompts.items() if k != "negative"}, "negative_prompt_chars": len(prompts["negative"])})
    print({"image": {**facts, **source}})
    print({"prepared_image": str(prepared_path.relative_to(run.root)), **crop})
    print({"contract": INPUT_SCHEMA["validation"]})

    probes = []
    for name, check in (
        ("empty prompt", lambda: validate_prompt("   ", name="T2V prompt")),
        ("image too small", lambda: validate_image(Image.new("RGB", (200, 150)))),
        ("48 frames (not 4k+1)", lambda: validate_settings(height=512, width=768, num_frames=48, steps=50, guidance_scale=5.0, seed=42)),
    ):
        try:
            check()
            probes.append({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            probes.append({"probe": name, "rejected": str(exc)[:140]})
        print(probes[-1])
    if any("rejected" not in p for p in probes):
        raise AssertionError(f"a refusal probe was accepted: {probes}")
    record = {"settings": settings, "prompts": prompts, "image_source": source, "image_facts": facts, "image_prepared": crop, "prepared_image": "i2v_conditioning.png", "probes": probes, "journey": run.tag or "default"}
    run.write_state("inputs.json", record)
    run.write_output("prepare.json", record)


def stage_encode(run: Run) -> None:
    """Section 5: encode the three prompts once with the 4-bit encoder, then release it before any transformer loads."""
    inputs = run.read_state("inputs.json", "encode")
    t0 = time.perf_counter()
    report = encode_prompts(run, inputs["prompts"])
    report.update({"seconds": round(time.perf_counter() - t0, 1), "host_rss_peak_gb": host_rss_peak_gb()})
    for key, value in report["embeddings"].items():
        print({key: value})
    print({k: v for k, v in report.items() if k != "embeddings"})
    run.write_output("encode.json", report)


def run_capability(run: Run, capability: str, *, seed: int | None = None, label: str | None = None) -> dict[str, Any]:
    """Generate one clip, check its decoded frames, write the frames, an mp4 and a record; return the record."""
    import numpy as np
    from PIL import Image

    from kandinsky5_video_pipeline import frame_checks, sha256_file, to_uint8

    inputs = run.read_state("inputs.json", capability)
    settings = dict(inputs["settings"])
    if seed is not None:
        settings["seed"] = seed
    label = label or capability
    image = Image.open(run.out / run.name(inputs["prepared_image"])).convert("RGB") if capability == "i2v" else None
    frames, info = generate_clip(run, capability, settings, image)
    frames_u8 = to_uint8(frames)
    report = frame_checks(frames, expected_frames=settings["num_frames"], height=settings["height"], width=settings["width"])
    print({"capability": capability, **info})
    print({"frame_checks": report["checks"], "consecutive_change": report["consecutive_change"], "blank_frames": report["blank_frames"], "frozen_pairs": report["frozen_pairs"]})
    if not (report["checks"]["all_values_finite"] and report["checks"]["no_blank_frames"]):
        raise RuntimeError(f"{label}: the decoded frames are not finite or are blank ({report['checks']}); this is the float16 overflow failure described in 'What did not work'. See Troubleshooting")
    np.save(run.state / run.name(f"{label}_frames.npy"), frames_u8)
    video_path = run.out / run.name(f"{STEM}_{label}.mp4")
    write_video(frames_u8, video_path, settings["fps"])
    strip = np.concatenate([frames_u8[i] for i in (0, len(frames_u8) // 2, len(frames_u8) - 1)], axis=1)
    strip_path = run.out / run.name(f"{STEM}_{label}_strip.jpg")
    Image.fromarray(strip).resize((strip.shape[1] // 2, strip.shape[0] // 2)).save(strip_path, quality=90)
    record = {"capability": capability, "label": label, "settings": settings, "generation": info, "frame_checks": report, "video": {"file": video_path.name, "bytes": video_path.stat().st_size, "sha256": sha256_file(video_path)}, "strip": strip_path.name, "host_rss_peak_gb": host_rss_peak_gb()}
    print({"video": record["video"], "strip": strip_path.name})
    run.write_output(f"{label}.json", record)
    return record


def write_video(frames_u8: Any, path: Path, fps: int) -> None:
    """Encode uint8 frames as H.264 mp4 with diffusers' exporter (imageio + imageio-ffmpeg backend).

    diffusers 0.40.0 multiplies numpy frames by 255 (it expects floats in 0..1), so uint8 arrays would overflow; PIL
    images are passed through unscaled.
    """
    from diffusers.utils import export_to_video
    from PIL import Image

    export_to_video([Image.fromarray(frame) for frame in frames_u8], str(path), fps=fps)


def selected_capabilities(run: Run) -> tuple[str, ...]:
    """The capabilities this journey generated, in canonical order (the default path runs both)."""
    caps = tuple(c for c in CAPABILITIES if c in {s.strip() for s in run.options.capabilities.split(",")})
    if not caps:
        raise ValueError(f"--capabilities={run.options.capabilities!r} names neither 't2v' nor 'i2v'")
    return caps


def stage_t2v(run: Run) -> None:
    """Section 6: text-to-video with the sft transformer."""
    run_capability(run, "t2v")


def stage_i2v(run: Run) -> None:
    """Section 7: image-to-video with the I2V transformer, conditioned on the prepared image."""
    run_capability(run, "i2v")


def stage_evaluate(run: Run) -> None:
    """Section 8: decoded-frame checks, first-frame fidelity and the CLIP prompt-alignment proxy with a swapped-prompt baseline."""
    import numpy as np
    from PIL import Image

    from kandinsky5_video_pipeline import alignment_summary, first_frame_fidelity

    inputs = run.read_state("inputs.json", "evaluate")
    caps = selected_capabilities(run)
    clips = {cap: np.load(run.state / run.name(f"{cap}_frames.npy")) for cap in caps}
    records = {cap: json.loads((run.out / run.name(f"{cap}.json")).read_text(encoding="utf-8")) for cap in caps}
    conditioning = np.asarray(Image.open(run.out / run.name(inputs["prepared_image"])).convert("RGB"))
    t0 = time.perf_counter()
    sims = clip_similarities(run, clips, inputs["prompts"], conditioning)
    alignment = {cap: alignment_summary(sims[cap]["own"], sims[cap]["other"]) for cap in caps}
    evaluation = {
        "capabilities": list(caps),
        "frame_checks": {cap: {"passed": records[cap]["frame_checks"]["passed"], **records[cap]["frame_checks"]["checks"]} for cap in caps},
        "prompt_alignment": alignment,
        "per_frame": sims,
        "scorer": "CLIP ViT-L/14 (the text_encoder_2 weights of the pinned T2V snapshot, vision tower included); no extra download",
        "evidence_level": "uncalibrated tutorial evidence: sanity checks and a CLIP proxy on one clip per capability, compared only with a swapped-prompt baseline; not a benchmark and not a measure of video quality",
        "calibration": "none: frame-check thresholds were chosen from development failures, not fitted to human judgements; CLIP cosine similarities are not probabilities",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if "i2v" in clips:
        to_image = np.asarray(sims["i2v_to_conditioning_image"], dtype=float)
        evaluation["i2v_first_frame_fidelity"] = first_frame_fidelity(clips["i2v"][0], conditioning)
        evaluation["i2v_clip_similarity_to_conditioning_image"] = {"first_frame": round(float(to_image[0]), 4), "last_frame": round(float(to_image[-1]), 4), "mean": round(float(to_image.mean()), 4)}
    for cap in caps:
        print({cap: {"frame_checks_passed": evaluation["frame_checks"][cap]["passed"], **{k: v for k, v in alignment[cap].items() if k != "reading"}}})
    if "i2v" in clips:
        print({"i2v_first_frame_fidelity": evaluation["i2v_first_frame_fidelity"], "i2v_clip_similarity_to_conditioning_image": evaluation["i2v_clip_similarity_to_conditioning_image"]})
    run.write_output("evaluation.json", evaluation)


def stage_report(run: Run) -> None:
    """Section 9: write the provenance record (pins, digests, seeds, settings, runtime, versions, outputs) and list the outputs."""
    from kandinsky5_video_pipeline import (
        COMPONENT_LICENSES,
        I2V_ID,
        I2V_REVISION,
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_REVISION,
        SAMPLE_IMAGE_URL,
        sha256_file,
    )

    inputs = run.read_state("inputs.json", "report")
    source_path = run.root / "source.json"
    notebook_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    caps = selected_capabilities(run)
    records = {cap: json.loads((run.out / run.name(f"{cap}.json")).read_text(encoding="utf-8")) for cap in caps}
    encode = json.loads((run.out / run.name("encode.json")).read_text(encoding="utf-8"))
    evaluation = json.loads((run.out / run.name("evaluation.json")).read_text(encoding="utf-8"))
    manifests = {key: {"sha256": sha256_file(run.snapshot(key) / MANIFEST_NAME), "verified_in_stage": "weights"} for key in SNAPSHOTS}
    provenance = {
        "notebook_source": notebook_source,
        "repository_revision": notebook_source["revision"] if notebook_source else None,
        "journey": inputs["journey"],
        "models": {
            "t2v": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "i2v": {"id": I2V_ID, "revision": I2V_REVISION, "shared_components_from": MODEL_ID},
            "licenses": COMPONENT_LICENSES,
            "manifests": manifests,
            "safetensors_only": True,
            "remote_code_executed": False,
            "text_encoder_quantization": "bitsandbytes 4-bit NF4, float16 compute (Qwen2.5-VL); CLIP text encoder float16",
            "generation_precision": "float16 transformer and VAE; VAE tiling on",
        },
        "inputs": {"prompts": inputs["prompts"], "image_source": inputs["image_source"], "image_prepared": inputs["image_prepared"], "sample_image_url": SAMPLE_IMAGE_URL if inputs["image_source"]["kind"] == "sample" else None},
        "settings": inputs["settings"],
        "runtime": {**runtime_versions(), "environment": "isolated hash-locked environment (one process per stage)"},
        "capabilities": list(caps),
        "timings_seconds": {"encode": encode["seconds"], **{cap: records[cap]["generation"]["generate_seconds"] for cap in caps}},
        "memory": {"encode_host_rss_peak_gb": encode["host_rss_peak_gb"], "encode_gpu_peak_gb": encode["gpu_peak_gb"], **{f"{cap}_gpu_peak_gb": records[cap]["generation"]["gpu_peak_gb"] for cap in caps}},
        "outputs": {cap: records[cap]["video"] for cap in caps},
        "evaluation_summary": {"frame_checks": evaluation["frame_checks"], "prompt_alignment": {cap: {k: v for k, v in evaluation["prompt_alignment"][cap].items() if k != "reading"} for cap in caps}, "i2v_first_frame_fidelity": evaluation.get("i2v_first_frame_fidelity")},
        "evidence_level": evaluation["evidence_level"],
        "calibration": evaluation["calibration"],
    }
    path = run.write_output(f"{STEM}_provenance.json", provenance)
    print({"provenance": str(path.relative_to(run.root)), "journey": provenance["journey"], "videos": {cap: provenance["outputs"][cap]["file"] for cap in caps}})
    print("outputs/:")
    for item in sorted(run.out.rglob("*")):
        if item.is_file():
            print(f"  - {item.relative_to(run.root).as_posix()} ({item.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """Section 10 (optional): change one thing - the seed - and regenerate the text-to-video clip."""
    import numpy as np

    first = np.load(run.state / run.name("t2v_frames.npy")).astype(np.float32) / 255.0
    record = run_capability(run, "t2v", seed=run.options.activity_seed, label=f"t2v_seed{run.options.activity_seed}")
    second = np.load(run.state / run.name(f"t2v_seed{run.options.activity_seed}_frames.npy")).astype(np.float32) / 255.0
    difference = {"mean_abs_pixel_difference_between_seeds": round(float(np.abs(first - second).mean()), 4), "seeds": [run.read_state("inputs.json", "activity")["settings"]["seed"], run.options.activity_seed]}
    print(difference)
    run.write_output("activity.json", {**difference, "clip": record["video"]})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "encode": stage_encode,
    "t2v": stage_t2v,
    "i2v": stage_i2v,
    "evaluate": stage_evaluate,
    "report": stage_report,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from kandinsky5_video_pipeline import GUIDANCE_SCALE, HEIGHT, I2V_PROMPT, NUM_FRAMES, SEED, STEPS, T2V_PROMPT, WIDTH

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the two snapshots and the sample image")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--tag", default="", help="journey tag prefixed to state and output names (BYOD uses 'byod')")
    parser.add_argument("--t2v-prompt", default=T2V_PROMPT)
    parser.add_argument("--i2v-prompt", default=I2V_PROMPT)
    parser.add_argument("--image", default="", help="prepare: an image file inside the runtime instead of the pinned sample")
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--activity-seed", type=int, default=ACTIVITY_SEED)
    parser.add_argument("--capabilities", default="t2v,i2v", help="evaluate/report: the capabilities this journey generated")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # The carried package lives under <root>/src; it must be importable before the full parser reads its defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--root", type=Path, required=True)
    carried_src = pre.parse_known_args(argv)[0].root.resolve() / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    options = parse_args(argv)
    run = Run(options.root.resolve(), options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1), "host_rss_peak_gb": host_rss_peak_gb()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
