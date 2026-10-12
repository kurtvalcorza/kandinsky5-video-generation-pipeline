---
license: other
license_name: mit-and-tencent-hunyuan-community
model_card_spec: "1.2"
pipeline_tag: text-to-video
base_model: kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
date_published: "2025-10-01"
---

# Kandinsky 5.0 Video Lite (T2V sft 5s + I2V 5s) — DIMER pipeline

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kurtvalcorza/kandinsky5-video-generation-pipeline/blob/main/tutorials/kandinsky5_video_generation_colab.ipynb)
[![Hugging Face T2V](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kandinsky--5.0--T2V--Lite--sft--5s-ffcc4d?style=flat)](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/tree/5c74c11b3ecfb96249b588d17aebd59f3fd9a080)
[![Hugging Face I2V](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kandinsky--5.0--I2V--Lite--5s-ffcc4d?style=flat)](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers/tree/8d965626ab76734922b00e359ca2d8faab2e0417)
[![Upstream](https://img.shields.io/badge/Upstream-kandinskylab%2Fkandinsky--5-181717?style=flat&logo=github&logoColor=white)](https://github.com/kandinskylab/kandinsky-5)
[![Paper](https://img.shields.io/badge/arXiv-2511.14993-b31b1b.svg)](https://arxiv.org/abs/2511.14993)
[![VAE licence](https://img.shields.io/badge/VAE-Tencent%20Hunyuan%20Community-blue.svg)](https://huggingface.co/tencent/HunyuanVideo/blob/6204ad6aea1a77ff5aba337c88278bb9500eb37d/LICENSE)

> **⚠️ Notice.** For research, training and evaluation only. Not validated for production, clinical, diagnostic or
> safety-critical use. The pipeline downloads third-party model weights under their own licences (below). Generated
> videos are synthetic and must not be presented as footage of real events. The bundled HunyuanVideo VAE is licensed
> only outside the European Union, the United Kingdom and South Korea. Everything is provided "as is", without warranty.

## Interactive Colab Tutorials

| Notebook | Kind | Profile | Mode | Runtime | Status |
|---|---|---|---|---|---|
| [`tutorials/kandinsky5_video_generation_colab.ipynb`](tutorials/kandinsky5_video_generation_colab.ipynb) | `TUTORIAL` | `MULTI-CAPABILITY` | `GUIDED` | Kaggle T4 only | Candidate — default Kaggle T4 run recorded at `ae58749`; BYOD journey not yet run |

The filename follows the fleet's `<model>_<task>_colab.ipynb` pattern; the notebook itself supports Kaggle only, because
free Colab sessions do not have the host memory the 4-bit encoder load needs.

#### Description

This repository packages two pinned Kandinsky 5.0 Video Lite checkpoints from Kandinsky Lab: the supervised fine-tuned
text-to-video model `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` at revision
`5c74c11b3ecfb96249b588d17aebd59f3fd9a080`, and the image-to-video model `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers`
at revision `8d965626ab76734922b00e359ca2d8faab2e0417`. Each is a diffusion transformer of about 2 billion parameters
(4,573,130,528 bytes of float weights per transformer file) that denoises a video latent with a flow-matching sampler under
classifier-free guidance. Both read prompts through Qwen2.5-VL-7B-Instruct and the OpenAI CLIP ViT-L/14 text tower, and
decode latents with the HunyuanVideo 3D VAE; these shared components are byte-identical in the two repositories at the
pinned revisions, so one 28,442,425,630-byte download set serves both. The repository's notebook runs both capabilities on a
Kaggle T4 by loading the 7B encoder in 4-bit NF4 and releasing it before generation, and by running the transformers and the
VAE in float16 with VAE tiling. The repository adds input validation, digest-verified staging, decoded-frame checks, a CLIP
prompt-alignment proxy and a provenance record. It does not train or modify the models.

#### Intended Use and Limitations

###### Primary Intended Uses

The intended use is learning and research: running a modern open video diffusion pipeline on a single free-tier 16 GB GPU,
and drafting short illustrative clips for training material, for example a five-second scene for a lecture on coastal or
volcanic hazards. Each capability produces 49 frames of 512 × 768 pixels, about two seconds at 24 frames per second, from a
prompt (text-to-video) or from a prompt and a photograph (image-to-video). Outputs are drafts for a human to review. The
notebook's checks tell a user whether a clip decoded correctly and moved, not whether it is fit for an audience.

###### Primary Intended Users

The intended users are students, lecturers and technical staff who can run a hosted notebook and read short Python, and who
want to understand how a text-to-video or image-to-video pipeline is assembled, constrained to a small GPU and checked. They
need a Kaggle account with GPU quota; one run of the notebook uses about an hour of T4 time (an estimate, see Environment).
Users who need production video generation, longer clips or higher resolutions are not served by this repository.

###### Out-of-scope use cases

Out of scope: any use inside the European Union, the United Kingdom or South Korea, where the bundled VAE's licence does not
apply; presenting generated clips as real footage for news, disaster response, legal evidence or public information;
depicting real, identifiable people; content that the Tencent Hunyuan Acceptable Use Policy or applicable law prohibits;
services with more than 100 million monthly active users without a separate licence from Tencent; safety-critical or
automated decision making; and any reading of the CLIP proxy or the frame checks as a measure of video quality. The
distilled 16-step checkpoints are out of scope on the T4 because they overflow in float16 (see Approaches to uncertainty).

#### Factors

###### Groups

The models generate people, places and cultures from text, so their outputs can reflect the demographic, geographic and
cultural balance of their web-scale training data, which this repository does not document or measure. Prompts about
Philippine settings, languages or people may be rendered less faithfully than prompts about settings common in the
training data; this is not measured here. The default sample photograph shows a Philippine landscape with distant,
unidentifiable people. No group-level evaluation exists in this repository.

###### Instrumentation

Inputs are free text and, for image-to-video, a JPEG, PNG or WEBP photograph. Validation refuses prompts outside 3..1000
characters, images under 256 pixels on the shorter side or over 8192 on the longer side, images over 25 MiB, and aspect
ratios outside 0.5..2.5. Images are resized to cover 768 × 512 and centre-cropped, and the crop is reported. Prompts longer
than 512 Qwen tokens or 77 CLIP tokens are truncated by the encoders, which the encode stage reports. Camera, compression
and watermarks in a conditioning photograph carry into the generated clip.

###### Environment

The supported environment is a Kaggle notebook with a Tesla T4 (15,360 MiB, compute capability 7.5) and at least 24 GB of
available host RAM, in a Kaggle notebook session (`KAGGLE_KERNEL_RUN_TYPE` and `/kaggle`); the notebook stops elsewhere. Stages run in an isolated environment of CPython 3.12.12 built from a
hash lock of torch 2.11.0+cu128 and torchvision 0.26.0+cu128 (PyTorch CUDA 12.8 index, the build the probe ran), diffusers 0.40.0, transformers 5.16.1, accelerate 1.14.0 and
bitsandbytes 0.50.2. Timing and memory figures in this card come from a development probe on a Kaggle T4 (2026-10-11/12,
Python 3.13, torch 2.11.0+cu128, in-kernel install), not from the notebook: about 150 s to download, 70 s to load the
encoders, 1352 s (text-to-video) and 1378 s (image-to-video) to generate, a 10.19 GB GPU peak during generation and an
18.6–21.9 GB host-memory peak while the 4-bit encoder loads.

#### Metrics

###### Performance Measures

The repository measures no quality metric. Its notebook reports, per clip: decoded-frame checks (all values finite; no
frame with pixel standard deviation below 0.01; mean change between neighbouring frames at least 0.002; no neighbouring
pair changing by more than 0.25); for image-to-video, the mean absolute difference, PSNR and CLIP image similarity between
the first frame and the conditioning photograph; and the mean CLIP ViT-L/14 cosine similarity of the frames to their own
prompt and to the other capability's prompt, with the margin between them. These are uncalibrated tutorial evidence: sanity checks and a coarse proxy, compared only with a swapped-prompt baseline, on one
clip per capability. Upstream benchmark results in the Kandinsky 5.0 report are not reproduced here.

###### Decision thresholds

The only thresholds are the uncalibrated frame-check limits listed under Performance Measures, chosen to catch the blank and frozen
outputs seen during development. A clip with non-finite or blank frames is refused; the motion and cut checks are reported
but do not stop the run. The CLIP margin has no threshold: its sign is reported as an observation. None of these thresholds
is calibrated against human judgement, and a deployment that needs an accept/reject decision would have to set and validate
its own threshold.

###### Approaches to uncertainty and variability

Each capability is run once with seed 42, so every reported number is one observation. Non-deterministic GPU kernels, the
4-bit quantization of the encoder, float16 rounding and library versions can change pixels for the same seed, and the
notebook's optional activity regenerates the text-to-video clip with seed 7 to show how much the starting noise alone
changes a clip. CLIP similarities are uncalibrated and are not probabilities. In development, the distilled 16-step
text-to-video checkpoint overflowed in float16 (48.5 % of the latent non-finite at step 12) and bfloat16 ran out of memory
on the T4; float16 is used and the latent's finiteness is tracked at every step.

#### Ethical considerations and biases

###### Data

The Kandinsky 5.0 report describes the training data at a high level; this repository does not inspect it and cannot rule
out copyrighted, personal or harmful material in it. The default inputs are two prompts written for the notebook, the
negative prompt from the upstream cards, and one CC0 photograph from Wikimedia Commons (`Mayon Volcano, Albay (Toe ).jpg`,
P.A.V. Mahusay, SHA-256 `138f5ff946e1e5a228270b838beacaa882983c2fea4ffb7e82bf4eefa1df1c01`). The photograph may have been
part of the models' training data. User prompts and images stay in the user's runtime and are not sent anywhere.

###### Human Life

The pipeline makes no decision about people and must not be used for one. Its risk to human life is indirect: a realistic
synthetic clip of a disaster, an injury or a public figure could mislead people if it is shown as real, for example during an
emergency. The notebook therefore labels every output as synthetic, forbids news, disaster-response and evidential use, and
the validators keep generation to short, low-resolution drafts.

###### Mitigations

The repository pins exact revisions and verifies every weight file by SHA-256 before loading, refuses pickle-capable
formats, executes no Hub-hosted code, needs no credential, and keeps user data inside the runtime. Inputs are validated
before any model loads. The notebook states the licence territory, the synthetic-media restriction and the evidence level of
every number. It does not filter prompts or outputs for unsafe content; that responsibility stays with the user and with the
Tencent Hunyuan Acceptable Use Policy, which applies to the VAE's outputs.

###### Risks and harms

Generated video can be used to deceive, to depict real people without consent, or to produce content that is harmful or
infringing; no safety filter is applied here. Using the pipeline inside the EU, the UK or South Korea would breach the VAE
licence. The models may reproduce biases or training material. The CLIP proxy can be gamed and does not detect harmful
content. A user who reads a passing frame check as a quality endorsement could publish a misleading clip.

###### Use cases

Suitable: classroom demonstrations of video diffusion, short illustrative drafts for training material reviewed by a human,
and experiments on how prompts, seeds and conditioning images change a clip. Not suitable: news or evidential footage,
depictions of real people, public-information content during emergencies, production services, or any use outside the VAE
licence territory. A new use should be checked against every licence in the table below before it starts.

## Licences

| Component | Source at the pinned revision | Licence |
|---|---|---|
| T2V and I2V diffusion transformers, scheduler configs | `kandinskylab/Kandinsky-5.0-*-Lite-*-Diffusers` (card metadata `license: mit`; upstream source repository `kandinskylab/kandinsky-5`, MIT) | MIT |
| HunyuanVideo 3D VAE (`vae/`) | bundled in both repositories; terms from `tencent/HunyuanVideo` `LICENSE` at revision `6204ad6aea1a77ff5aba337c88278bb9500eb37d` (SHA-256 `6ba8ea0dc1c464c9ab549b281be5d3afcab93c3308a6967d22240150d844c54d`) | Tencent Hunyuan Community License: territory excludes the EU, the UK and South Korea; >100 M monthly active users needs a separate licence; Acceptable Use Policy (Exhibit A) |
| Qwen2.5-VL-7B-Instruct text encoder (`text_encoder/`) | bundled; its card declares `license: apache-2.0` (upstream `Qwen/Qwen2.5-VL-7B-Instruct` at `cc594898137f460bfe9f0759e9844b3ce807cfb5`) | Apache-2.0 |
| CLIP ViT-L/14 (`text_encoder_2/`, text and vision towers) | bundled; its README is the OpenAI CLIP model card; upstream `openai/clip-vit-large-patch14` carries no licence metadata; the `openai/CLIP` source repository is MIT | MIT |
| Default sample photograph | Wikimedia Commons, CC0 1.0 | CC0 1.0 |
| This repository's code | `LICENSE` | Apache-2.0 |

The Kandinsky model cards declare MIT but do not mention the Tencent licence of the bundled VAE; this card states both.

## Immutable provenance

| Snapshot | Revision | Files | Bytes |
|---|---|---|---|
| `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` (all shared components + T2V transformer) | `5c74c11b3ecfb96249b588d17aebd59f3fd9a080` | 28 | 23,869,293,940 |
| `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers` (I2V transformer, scheduler, pipeline index) | `8d965626ab76734922b00e359ca2d8faab2e0417` | 4 | 4,573,131,690 |

Every file's size and SHA-256 is recorded in `weights/*/dimer-base-manifest.json`. `text_encoder_2/pytorch_model.bin`,
`flax_model.msgpack` and `tf_model.h5` (duplicates of the CLIP safetensors) and the repositories' assets are not staged.
`date_published` is the creation date of the T2V Diffusers repository on the Hugging Face Hub (first commit, 2025-10-01).

## Input/output contract

| Capability | Input | Output |
|---|---|---|
| Text-to-video | prompt, 3..1000 characters | 49 RGB frames, 512 × 768, H.264 mp4 at 24 fps; a frame-check record |
| Image-to-video | prompt + JPEG/PNG/WEBP image (≤ 25 MiB, shorter side ≥ 256 px, aspect 0.5..2.5), resized and centre-cropped to 768 × 512 | as above |

Settings are limited to the measured T4 envelope: height × width ≤ 512 × 768 (multiples of 16), `num_frames` = 4k+1 ≤ 49,
1..50 steps, guidance 1.0..10.0. Defaults: 50 steps, guidance 5.0, seed 42, the upstream negative prompt.

## Verification records

No execution of the notebook has been recorded; see `docs/release-verification.md`. Static checks (generator parity,
validators, lock audit, CPU unit tests) are not execution evidence.

## References

1. V. Arkhipkin et al., "Kandinsky 5.0: A family of foundation models for image and video generation," arXiv:2511.14993, 2025. https://arxiv.org/abs/2511.14993
2. W. Kong et al., "HunyuanVideo: A systematic framework for large video generative models," arXiv:2412.03603, 2024. https://arxiv.org/abs/2412.03603
3. S. Bai et al., "Qwen2.5-VL technical report," arXiv:2502.13923, 2025. https://arxiv.org/abs/2502.13923
4. A. Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021. https://arxiv.org/abs/2103.00020
5. Tencent, "Tencent Hunyuan Community License Agreement." https://huggingface.co/tencent/HunyuanVideo/blob/6204ad6aea1a77ff5aba337c88278bb9500eb37d/LICENSE
