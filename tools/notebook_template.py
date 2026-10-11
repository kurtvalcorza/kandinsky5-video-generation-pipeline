"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.4 §4 standalone, §8.1 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures a MULTI-CAPABILITY / GUIDED / TUTORIAL notebook: Kandinsky 5.0 Video Lite text-to-video (the
sft checkpoint) and then image-to-video, sharing one 4-bit text encoder pass and one VAE, on a Kaggle T4.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "kandinsky5-video-generation-pipeline"
NOTEBOOK = "kandinsky5_video_generation_colab.ipynb"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Hugging Face T2V", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kandinsky--5.0--T2V--Lite--sft--5s-ffcc4d?style=flat", "https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/tree/5c74c11b3ecfb96249b588d17aebd59f3fd9a080"),
    ("Hugging Face I2V", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kandinsky--5.0--I2V--Lite--5s-ffcc4d?style=flat", "https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers/tree/8d965626ab76734922b00e359ca2d8faab2e0417"),
    ("Paper", "https://img.shields.io/badge/arXiv-2511.14993-b31b1b.svg", "https://arxiv.org/abs/2511.14993"),
    ("License", "https://img.shields.io/badge/License-MIT%20%2B%20Tencent%20Hunyuan%20Community-blue.svg", f"https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md"),
]

NOTICE = (
    "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
    "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
    "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
    "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
    "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
    "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
    "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
    "> its use.\n"
    ">\n"
    "> **Synthetic media.** The videos this notebook generates are synthetic. They are not records of real events and must\n"
    "> not be presented as footage for news, disaster-response or evidential use. **Licence territory.** The bundled\n"
    "> HunyuanVideo VAE is licensed under the Tencent Hunyuan Community License, which does not apply in the European Union,\n"
    "> the United Kingdom or South Korea; do not run this notebook or use its outputs there."
)

TEMPLATE = {
    "package": "kandinsky5_video_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky5_video",
    "notebook_name": NOTEBOOK,
    "profile": "MULTI-CAPABILITY",
    "mode": "GUIDED",
    "kind": "TUTORIAL",
    "notice": NOTICE,
    "run_all": (
        "Select **Run all** in a fresh **Kaggle** notebook with the **GPU T4** accelerator and Internet on. The notebook stops "
        "in Section 1 with an actionable message on any other runtime: it needs a T4-class GPU and at least 24 GB of free host "
        "memory, which free Colab (about 12.7 GB) does not have. It builds an isolated Python environment from the carried "
        "hash-locked requirements without touching the kernel's packages, stages and SHA-256-verifies two pinned Hub snapshots "
        "(~{total_gb} GB), fetches and verifies one CC0 photograph, validates the prompts, the image and the settings, encodes "
        "the three prompts once with a 4-bit text encoder and releases it, generates a 49-frame text-to-video clip and then a "
        "49-frame image-to-video clip one model at a time, checks the decoded frames, scores prompt alignment with CLIP against a "
        "swapped-prompt baseline, and writes two mp4 files and a provenance record. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload, no configuration edit and no runtime restart. "
        "**Duration (estimate):** about one hour. The estimate is assembled from a development probe on a Kaggle T4 "
        "(2026-10-11/12; a different, in-kernel install on Python 3.13): about 150 s to download, about 70 s to load the "
        "encoders, 1352 s to generate the text-to-video clip and 1378 s for the image-to-video clip. The environment build and "
        "the SHA-256 verification of the downloads were not measured. This revision of the notebook has not yet been run."
    ),
    "byod": (
        "After the default path completes, Section 11 lets you generate from your own prompts and your own image with the same "
        "validation, encoding, generation, checks and export. Set `USE_BYOD = True`, write your prompts, and set "
        "`BYOD_IMAGE_PATH` to an image file already in the runtime (on Kaggle, add it as a dataset under `/kaggle/input/`). "
        "The image contract (JPEG, PNG or WEBP, at most 25 MiB, shorter side at least 256 px, width/height between 0.5 and 2.5) "
        "and the privacy guidance are stated in the Prerequisites and in Section 11. Your files stay inside this runtime; "
        "nothing is uploaded anywhere. BYOD is optional and never part of the default path."
    ),
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/kandinsky5_video_pipeline/__init__.py": "src/kandinsky5_video_pipeline/__init__.py",
        "src/kandinsky5_video_pipeline/pipeline.py": "src/kandinsky5_video_pipeline/pipeline.py",
        "src/kandinsky5_video_pipeline/inputs.py": "src/kandinsky5_video_pipeline/inputs.py",
        "src/kandinsky5_video_pipeline/checks.py": "src/kandinsky5_video_pipeline/checks.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-kaggle.lock.txt",
        "weights/kandinsky5-t2v-lite-sft-5s/dimer-base-manifest.json": "weights/kandinsky5-t2v-lite-sft-5s/dimer-base-manifest.json",
        "weights/kandinsky5-i2v-lite-5s/dimer-base-manifest.json": "weights/kandinsky5-i2v-lite-5s/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    # Runtime gate (ENV4, ENV12, RUN11). The RAM floor is the largest host-memory peak measured while loading the 4-bit
    # encoder in the development probe (21.94 GB, ru_maxrss of the probe kernel) plus about 2 GB of headroom.
    "runtime_gate": {"min_gpu_mib": 15000, "min_compute_capability": 7.5, "min_available_ram_gb": 24.0, "measured_peak_ram_gb": 21.9, "environment_gb": 12},
    "runtime_modules": ["torch", "diffusers", "transformers", "bitsandbytes"],
    "title": "Kandinsky 5.0 Video Lite — DIMER guided notebook: text-to-video and image-to-video on a Kaggle T4 (standalone)",
    "badges": BADGES,
    "capability": "text-to-video and image-to-video generation with the 2B-parameter Kandinsky 5.0 Video Lite diffusion transformers, sharing one text-encoding pass and one video VAE",
    "intro": (
        "Kandinsky 5.0 [1] is a family of diffusion models for image and video generation. This notebook runs two of its "
        "**Video Lite** checkpoints, each a diffusion transformer of about 2 billion parameters: the supervised fine-tuned (sft) "
        "**text-to-video** model [2] and the **image-to-video** model [3]. Both read a prompt through two text encoders — "
        "Qwen2.5-VL-7B [5] for a long, detailed embedding and the CLIP ViT-L/14 text tower [6] for a pooled summary — denoise a "
        "video latent with a flow-matching sampler [11] under classifier-free guidance [12], and decode the latent to frames "
        "with the HunyuanVideo 3D VAE [4]. The two checkpoints share the encoders and the VAE byte for byte, so one download "
        "set and one encoding pass serve both capabilities.\n\n"
        "The run fits a 16 GB T4 only because of three choices, each explained where it happens: the 7B text encoder is loaded "
        "in **4-bit NF4** [8] and **released** before any transformer loads (Section 5); the transformers and the VAE run in "
        "**float16**, because the T4 has no fast bfloat16 path (Sections 6–7 and **What did not work**); and the VAE decodes in "
        "**tiles**. **Generated video has no ground truth**, so Section 8 reads sanity checks — are the frames finite, not "
        "blank, moving and free of cuts — and a CLIP prompt-alignment proxy compared with a swapped-prompt baseline. None of "
        "these is a judgement of video quality."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** how a prompt becomes a video in Kandinsky 5.0 (two text encoders → a diffusion transformer denoising a "
        "video latent under classifier-free guidance → a 3D VAE decoder), and why the encoders can be released before "
        "generation (Section 5).\n"
        "2. **Diagnose** a refused input from its validation message, and **identify** the operating limits of the T4 path "
        "(Section 4).\n"
        "3. **Compare** text-to-video and image-to-video: what each takes as input and what the conditioning image fixes "
        "(Sections 6–7).\n"
        "4. **Interpret** decoded-frame checks, first-frame fidelity and a CLIP prompt-alignment margin, and **distinguish** "
        "them from a measure of quality (Section 8).\n"
        "5. **Trace** a generated clip back to its model revisions, digests, seed and settings in the provenance record "
        "(Section 9).\n"
        "6. **Predict**, run and **explain** the effect of changing the seed in an optional activity (Section 10).\n"
        "7. **Write** an evidence-based conclusion that names what one clip per capability can and cannot show (Conclusion)."
    ),
    "exclusions": (
        "the distilled 16-step or 10-second checkpoints, the Pro models, text-to-image, fine-tuning or LoRA adaptation, "
        "prompt-safety filtering, human preference or VBench-style benchmarks, audio, and any claim that a CLIP score or a frame "
        "check measures video quality. Free Colab and CPU-only runtimes are not supported."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Kaggle** notebook with the **GPU T4** accelerator (T4 x2 is fine: the stages use the first GPU only) and Internet on. Section 1 refuses a GPU with less than 15,000 MiB or compute capability below 7.5, and a session with less than 24 GB of available host RAM. The kernel's Python version does not matter: the notebook installs nothing into it and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.11.0 from PyPI, whose Linux wheel is the CUDA 13 build). About {total_gb} GB of disk is needed for the weights (written to the temporary disk, not `/kaggle/working`) and about 12 GB for the environment. The lock targets manylinux x86_64 only; other platforms stop in Section 1.",
        "- **Precision and placement:** Qwen2.5-VL-7B is quantized to 4-bit NF4 by `bitsandbytes` with float16 compute and placed on the GPU; the CLIP text encoder runs in float16; both are released before generation. Each transformer and the VAE run in float16 on the GPU with VAE tiling on. Kernels are not forced to be deterministic, so the same seed can give slightly different pixels on another GPU or library build.",
        "- **Knowledge:** you can run notebook cells and read short Python. Helpful but not required: what a diffusion model does at inference, what classifier-free guidance is, and why a training-free check is not a quality score. Terms are explained where they are first used and collected in the Glossary.",
        "- **Weights:** every staged weight file is safetensors; `pytorch_model.bin`, `flax_model.msgpack` and `tf_model.h5` duplicates in the upstream repository are not downloaded, nothing is unpickled, and no Hub-hosted code is executed — the model classes come from `diffusers` [7] and `transformers` on PyPI. Licences: the Kandinsky 5.0 transformers are MIT; the HunyuanVideo VAE is under the Tencent Hunyuan Community License [10] (territory excludes the EU, the UK and South Korea; a separate licence is required above 100 million monthly active users; an Acceptable Use Policy applies); Qwen2.5-VL-7B is Apache-2.0; the OpenAI CLIP weights are MIT. The model card has the details.",
        "- **Data contract:** a prompt is 3..1000 characters of printable text (the Qwen encoder keeps at most 512 tokens and CLIP at most 77; Section 5 reports truncation). An image is JPEG, PNG or WEBP, at most 25 MiB, shorter side at least 256 px, longer side at most 8192 px, width/height between 0.5 and 2.5; it is resized to cover 768 × 512 px and centre-cropped, and the crop is reported. Settings stay inside the measured T4 envelope: at most 512 × 768 px, `num_frames` = 4k+1 up to 49, 1..50 steps, guidance 1.0..10.0.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing, and BYOD files stay inside the runtime.",
    ],
    "guided": {
        "opening": [
            (
                "## Problem statement\n\n"
                "Teams that prepare training and outreach material — for example a disaster-risk-reduction office explaining "
                "volcanic hazards, or a lecturer illustrating a coastal scene — sometimes need short illustrative video clips and "
                "have no footage or budget to film. Open text-to-video and image-to-video models can draft such clips, but they "
                "are large, and most practitioners in the Philippines work on free hosted GPUs rather than data-centre hardware. "
                "This notebook asks a narrow question: **can the two Kandinsky 5.0 Video Lite models produce a usable five-second "
                "draft on a single 16 GB T4, from a prompt alone and from a prompt plus a photograph, and how would you check the "
                "result without ground truth?**\n\n"
                "**What a useful outcome looks like.** Each capability returns 49 frames at 512 × 768 px that pass every "
                "decoded-frame check in Section 8 (finite, none blank, visible motion, no abrupt cuts); the image-to-video clip "
                "starts close to its conditioning photograph; and each clip's CLIP similarity to its own prompt exceeds its "
                "similarity to the other capability's prompt. These are the observable signals the notebook can read; whether "
                "a clip is good enough for a given audience is a human judgement the notebook does not make.\n\n"
                "**Scope boundary.** The notebook does not address photorealistic footage for news or evidence, depictions of real "
                "people, long or high-resolution video, audio, fine-tuning, or any setting outside the licence territory of the "
                "bundled VAE. Use cases it serves: drafting illustrative clips for training material, and learning how a video "
                "diffusion pipeline is assembled, measured and pinned."
            ),
            (
                "## Data context\n\n"
                "There is no dataset: the inputs are two prompts written for this notebook and one photograph.\n\n"
                "- **Prompts.** The text-to-video prompt describes an outrigger fishing boat at sunrise; the image-to-video prompt "
                "describes cloud and palm motion over the photograph's lava field. Both were written by the repository author for "
                "this notebook. The negative prompt is the one published with the Kandinsky 5.0 Lite model cards [2].\n"
                "- **Photograph.** *Mayon Volcano, Albay (Toe )* [9]: a 3264 × 1840 px JPEG (2,182,680 bytes) taken on 2016-08-18 "
                "at the toe of a lava flow of Mayon Volcano, Albay, by P.A.V. Mahusay, dedicated to the public domain under CC0 "
                "1.0 and hosted on Wikimedia Commons. The notebook pins it by URL, byte size and SHA-256 "
                "(`138f5ff9…df1c01`) and refuses it on any mismatch rather than substituting another image.\n"
                "- **Representativeness and gaps.** One photograph and two prompts exercise every stage but say nothing about how "
                "the models behave across subjects, styles or prompt lengths. The photograph has a small corner watermark and "
                "distant, unidentifiable people, both of which the image-to-video model may animate or distort. It is a public "
                "image, so it may have been part of the models' web-scale training data; the CLIP similarities could therefore be "
                "optimistic for this sample, and BYOD with your own photograph avoids that overlap."
            ),
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook, run cells in order and read short Python, and "
                "who want to see how a video diffusion pipeline is run, constrained to a small GPU and checked. No prior experience "
                "with video models is assumed. You need a Kaggle account with GPU quota: one run uses about an hour of T4 time.\n\n"
                "**Running it.** In Kaggle, open *Settings → Accelerator → GPU T4 x2* and turn *Internet* on, then choose *Run "
                "all*. The default path needs no edit, no upload, no token and no restart. Sections 1–3 build the environment and "
                "download about {total_gb} GB, and Sections 6 and 7 each take more than 20 minutes; read ahead while they run.\n\n"
                "**Where the code runs.** The kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files in the run directory, so GPU and host memory are "
                "returned to the system between models.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–11) are the workflow; each runs one stage and prints compact "
                "dictionaries. *Infrastructure cells* (Sections 1–3) are collapsed and titled **Infrastructure**: you may run them "
                "without studying their implementation.\n\n"
                "**Form controls.** `RUN_ACTIVITY` (Section 10) and `USE_BYOD`, `BYOD_IMAGE_PATH`, `BYOD_T2V_PROMPT`, "
                "`BYOD_I2V_PROMPT` and `BYOD_CAPABILITIES` (Section 11) are form fields. Leave them at their defaults for the first "
                "run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and read. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
                "expectation; after it, **What to notice** describes normal output, and a collapsed **Check your reasoning** answer "
                "follows. Exact numbers vary between runs and GPUs, so the notes describe the shape of a normal result."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Capability | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **A. Text-to-video** | a prompt (and the fixed negative prompt) | Qwen2.5-VL-7B (4-bit) + CLIP text encoder → prompt embeddings → sft diffusion transformer, 50 flow-matching steps under guidance 5.0 → HunyuanVideo VAE decoder | 49 RGB frames, 512 × 768 px, an mp4 at 24 fps (about 2 s) |\n"
                "| **B. Image-to-video** | a prompt + a photograph (resized and centre-cropped to 768 × 512) | the same embeddings path → I2V diffusion transformer conditioned on the encoded photograph → the same VAE | 49 RGB frames, 512 × 768 px, an mp4 at 24 fps |\n"
                "| **Checks** | both clips, both prompts, the photograph | numpy frame statistics; CLIP ViT-L/14 image and text towers from the same snapshot | a pass/fail per check, a prompt-alignment margin per clip, first-frame fidelity |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | GPU, host RAM and disk checked; a fresh run directory | the GPU and free RAM |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions, `cuda: True` |\n"
                "| 3. Pin, stage and verify | [Engineering] | two snapshots downloaded and SHA-256-checked | file counts |\n"
                "| 4. Inputs and refusals | [Evaluation practice] | prompts, photograph and settings validated; three refusals | the crop, the refusals |\n"
                "| 5. Encode once, release | [Concept] | three prompts encoded by the 4-bit encoder, which is then freed | memory before and after |\n"
                "| 6. Capability A: text-to-video | [Concept] | the sft transformer generates 49 frames | a frame strip, the checks |\n"
                "| 7. Capability B: image-to-video | [Concept] | the I2V transformer animates the photograph | a frame strip, the checks |\n"
                "| 8. Evaluate | [Evaluation practice] | frame checks, first-frame fidelity, CLIP alignment vs a swapped prompt | the principal result |\n"
                "| 9. Export and provenance | [Engineering] | mp4 files and a provenance record | file list, digests |\n"
                "| 10. Optional activity | [Concept] | change the seed and regenerate | how much the clip changes |\n"
                "| 11. Bring Your Own Data | [Evaluation practice] | your prompts and image through the same stages | your clips |\n\n"
                "A **fast path** exists only for reading: the frame strips in Sections 6–7 and the summary in Section 8 carry the "
                "principal results."
            ),
            (
                "## Glossary\n\n"
                "<details>\n<summary>Open the glossary</summary>\n\n"
                "- **Diffusion transformer (DiT):** a transformer that predicts how to move a noisy latent towards a clean one, one "
                "step at a time.\n"
                "- **Latent:** a compressed representation of the video; the VAE maps frames to latents and back. The HunyuanVideo "
                "VAE compresses 4 frames into 1 latent step in time and 8 × 8 pixels into 1 latent cell in space, which is why "
                "`num_frames` must be 4k+1.\n"
                "- **Flow matching:** the training objective and sampler family these models use; the sampler integrates a learned "
                "velocity field from noise to data over `steps` Euler steps [11].\n"
                "- **Classifier-free guidance (CFG):** each step runs the transformer twice — with the prompt and with the negative "
                "prompt — and moves the result away from the negative by a factor of `guidance_scale` [12]. It doubles the cost of "
                "a step.\n"
                "- **4-bit NF4 quantization:** stores weights in 4-bit NormalFloat blocks and de-quantizes them on the fly; it cuts "
                "the 7B encoder from about 16.6 GB to about 6.7 GB of GPU memory [8].\n"
                "- **float16 / bfloat16:** 16-bit number formats. float16 has more precision but a smaller range (largest value "
                "65,504), so large activations can overflow to infinity; bfloat16 has the range of float32 but is not fast on a T4.\n"
                "- **VAE tiling:** decoding the latent in overlapping spatial tiles so the decoder fits in memory.\n"
                "- **Seed:** the number that fixes the starting noise; the same seed, settings and software give the same clip on "
                "the same hardware.\n"
                "- **CLIP similarity:** the cosine of the angle between CLIP's embedding of a frame and of a text; higher means "
                "closer in CLIP's space. It is uncalibrated and is not a probability.\n"
                "- **Swapped-prompt baseline:** the same frames scored against the *other* capability's prompt; a clip that sits "
                "closer to its own prompt than to an unrelated one shows at least coarse prompt alignment.\n"
                "- **Infrastructure cell:** a collapsed setup cell for reproducibility and provenance, not a learning objective.\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. You may run it without studying its implementation.\n\n"
                "The cell stops with an actionable message unless it finds Linux x86_64 (the lock is built for manylinux x86_64), an "
                "NVIDIA GPU with at least 15,000 MiB and compute capability 7.5 (a T4 reports 15,360 MiB and 7.5), at least 24 GB of "
                "*available* host RAM, and enough disk. The RAM floor comes from development runs on a Kaggle T4: loading the "
                "4-bit text encoder peaked at 18.6–21.9 GB of host memory, because the encoder's float16 shards pass through host "
                "memory before they are quantized. Free Colab offers about 12.7 GB and cannot hold that peak, which is why this "
                "notebook supports Kaggle only. The cell also creates a fresh run directory under `outputs/` and puts the weights "
                "on the temporary disk, because Kaggle caps the working directory at about 20 GB."
            ),
            "after": (
                "**Expected result:** one dictionary naming a `Tesla T4`, its memory, compute capability `7.5`, the available host "
                "RAM (about 30 GB on Kaggle), `kaggle: True`, the run directory and the free disk."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. You may run them without studying their "
                "implementation.\n\n"
                "The first cell writes the files this notebook carries — the repository's small package (prompt, image and setting "
                "validation; pinned identities; frame checks), the stage runner, the hash-locked requirements, the two snapshot "
                "manifests and the licence — and checks each against its recorded SHA-256. The second downloads a pinned `uv`, builds "
                "an isolated Python 3.12 environment from the lock with `--require-hashes --only-binary :all:`, and defines "
                "`run_stage`, `load_record`, `show_image` and `show_video`, the helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and take a few minutes. "
                "A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, Python 3.12.12, the `torch`, `diffusers`, "
                "`transformers` and `bitsandbytes` versions, `torch_cuda` (the CUDA build of torch), `'cuda': True`, the number of "
                "locked packages and the setup time. If `'cuda'` is `False` the cell stops; see **Troubleshooting**."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the models · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about {total_gb} GB and checks every file's "
                "size and SHA-256; you may run it without studying its implementation.\n\n"
                "Two snapshots are pinned to immutable revisions, never a branch: `{MODEL_ID}` at `{MODEL_REVISION}` supplies the "
                "text-to-video transformer and every shared component (both text encoders, both tokenizers and the VAE), and "
                "`{I2V_ID}` at `{I2V_REVISION}` supplies only the image-to-video transformer, its scheduler config and its pipeline "
                "index. The shared files are byte-identical in both repositories at these revisions — the same 24 paths with the same "
                "SHA-256 — so they are downloaded once. The `weights` stage installs each carried manifest, fetches exactly the absent "
                "files from the Hub, and re-hashes every file, raising on the first size or digest mismatch. There is no fallback to "
                "another revision, and no remote code runs. Later stages re-check that every file is present at its recorded size "
                "before loading; the full SHA-256 check runs once, here, because re-hashing {total_gb} GB in every stage would add "
                "minutes each time."
            ),
            "after": (
                "**What to notice:** for each snapshot, its id, revision, file count and size; how many files were fetched (zero on a "
                "rerun in the same session); and the verified file count with the seconds the hashing took. A size or SHA-256 "
                "mismatch stops the cell with a `ValueError` naming the file — never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Inputs, validation and refusals · [Evaluation practice]\n\n"
                "From here on, every code cell runs one stage of the carried runner. The `prepare` stage fetches the pinned photograph "
                "(refusing it on any size or SHA-256 mismatch), validates both prompts, the photograph and the generation settings "
                "**before any model loads**, resizes the photograph to cover 768 × 512 px and centre-crops it, and reports exactly "
                "what the crop removed. It then feeds three deliberately broken inputs — an empty prompt, a 200 × 150 px image and "
                "48 frames — to the same validators to show that each is refused with a message naming the rule. **Validation** "
                "here is structural: nothing checks that a prompt is sensible or that the model can render it.\n\n"
                "**Predict before running:** the photograph is 3264 × 1840 px (about 16:9) and the model wants 768 × 512 (3:2). Will "
                "the crop remove picture from the top and bottom, or from the sides?"
            ),
            "code": "run_stage('prepare')\nshow_image('i2v_conditioning.png', 'the prepared 768 x 512 conditioning image')",
            "after": (
                "**What to notice:** the settings dictionary (512 × 768, 49 frames, 50 steps, guidance 5.0, seed 42, 24 fps); the "
                "photograph's facts and its pinned digest; the crop box and `fraction_of_resized_area_kept`; and three `rejected` "
                "lines, each naming its rule.\n\n"
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "The photograph is wider than 3:2, so scaling it to a height of 512 px makes it wider than 768 px, and the centre crop "
                "removes strips from the left and right. About 84 % of the resized area is kept. Anything at the edges of your own "
                "image is lost the same way, so keep the subject near the centre.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 5. Encode the prompts once, then release the encoder · [Concept]\n\n"
                "Both capabilities read a prompt the same way: Qwen2.5-VL-7B [5] turns it into a sequence of hidden states (up to "
                "512 tokens) and the CLIP text tower [6] into one pooled vector (up to 77 tokens). The 7B encoder alone would need "
                "about 16.6 GB in float16 — more than the T4 has — so the `encode` stage loads it in **4-bit NF4** [8], encodes the "
                "text-to-video prompt, the image-to-video prompt and the negative prompt once, saves the embeddings to a safetensors "
                "file, and exits. The encoders never coexist with a transformer on the GPU.\n\n"
                "**Predict before running:** after the encoders are released, how much GPU memory should still be allocated?"
            ),
            "code": "run_stage('encode')",
            "after": (
                "**What to notice:** an embedding shape for each prompt (Qwen: tokens × 3584; CLIP: 1 × 768) with `finite: True`; "
                "`clip_tokens` and `clip_truncated` (a prompt longer than 77 CLIP tokens loses its tail in the pooled vector, which is "
                "reported, not hidden); a GPU peak of roughly 7 GB; and close to 0 GB allocated after release. `host_rss_peak_gb` is "
                "the host-memory peak of this stage — the number the 24 GB gate in Section 1 protects.\n\n"
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "Close to zero: the stage deletes both encoders, empties the CUDA cache and then exits, and the process exit returns "
                "everything to the driver. That is why the 2B transformers in Sections 6 and 7 can use most of the 15 GB.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 6. Capability A — text-to-video · [Concept]\n\n"
                "The `t2v` stage loads the sft transformer [2] and the HunyuanVideo VAE [4] in **float16**, turns VAE tiling on, and "
                "generates 49 frames at 512 × 768 from the cached embeddings: 50 flow-matching steps [11], each running the "
                "transformer twice for classifier-free guidance at 5.0 [12], with seed 42. A callback records, at every step, the "
                "fraction of the latent that is still finite — the signal that exposed the float16 overflow of the distilled "
                "checkpoint (see **What did not work**). The stage refuses to write a clip whose frames are non-finite or blank.\n\n"
                "**Input contract:** one prompt. **Output contract:** `outputs/kandinsky5_video_t2v.mp4` (H.264, 24 fps), a three-frame "
                "strip (first, middle, last), and `outputs/t2v.json` with the settings, timings, GPU peak and frame checks.\n\n"
                "**Predict before running:** 50 steps with guidance means 100 transformer passes. If one pass over a 49-frame latent "
                "takes about 13–14 s on a T4, roughly how long will the generation take?"
            ),
            "code": "run_stage('t2v')\nshow_image('kandinsky5_video_t2v_strip.jpg', 'text-to-video: first, middle and last frame')",
            "after": (
                "**What to notice:** `generate_seconds` (in the development probe, 1352 s on a Kaggle T4); `first_nonfinite_step: "
                "None`; a GPU peak near 10 GB; every frame check `True`; and a strip whose three frames show the same scene with "
                "visible change between them. Run `show_video('kandinsky5_video_t2v.mp4')` in a new cell to play the clip.\n\n"
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "About 100 × 13.5 s ≈ 22–23 minutes, plus a few seconds of VAE decoding. Guidance doubles the work of every step; that "
                "is the main cost of a 50-step CFG sampler compared with a distilled 16-step one.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 7. Capability B — image-to-video · [Concept]\n\n"
                "The `i2v` stage loads the image-to-video transformer [3] from its own snapshot and the same VAE, encodes the prepared "
                "photograph into the latent space as the first frame's condition, and generates 49 frames with the same sampler, "
                "guidance, seed and negative prompt. The prompt now describes *motion* in a scene the photograph already fixes. The "
                "text-to-video transformer was released when Section 6's process exited, so only one transformer is ever in memory.\n\n"
                "**Input contract:** one prompt and one prepared 768 × 512 image. **Output contract:** "
                "`outputs/kandinsky5_video_i2v.mp4`, a strip and `outputs/i2v.json`, as in Section 6.\n\n"
                "**Predict before running:** which will change more from the first frame to the last — this clip or the "
                "text-to-video clip? Why?"
            ),
            "code": "run_stage('i2v')\nshow_image('kandinsky5_video_i2v_strip.jpg', 'image-to-video: first, middle and last frame')",
            "after": (
                "**What to notice:** a first frame that looks like the prepared photograph; the frame checks; and "
                "`consecutive_change`, the mean absolute change between neighbouring frames, to compare with Section 6's. In the "
                "development probe the image-to-video generation took 1378 s on a Kaggle T4.\n\n"
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "Usually the text-to-video clip: nothing anchors its first frame, and its prompt describes a moving boat and a "
                "tracking camera. The image-to-video clip is anchored to the photograph and its prompt asks for slow cloud and palm "
                "motion. Section 8 measures this with `consecutive_change`; one pair of clips cannot show it is a general rule.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 8. Evaluate both clips · [Evaluation practice]\n\n"
                "Generated video has no reference answer here, so the `evaluate` stage reads three kinds of evidence, each with a "
                "stated threshold or baseline:\n\n"
                "1. **Decoded-frame checks** (computed in Sections 6–7): every value finite; no blank frame (per-frame pixel standard "
                "deviation below 0.01 on a 0–1 scale); the clip moves (mean change between neighbouring frames at least 0.002); and "
                "no abrupt cut (no neighbouring pair changing by more than 0.25). These catch the failure modes seen in development "
                "— blank or frozen output — and say nothing about whether the video is good.\n"
                "2. **First-frame fidelity** (image-to-video only): mean absolute difference and PSNR between the first generated "
                "frame and the prepared photograph, plus their CLIP image similarity. A high PSNR says the model kept the "
                "photograph; it does not say the motion is plausible.\n"
                "3. **Prompt alignment with a swapped-prompt baseline:** CLIP ViT-L/14 [6] — the image and text towers stored in the "
                "same `text_encoder_2` file the pipeline already verified, so nothing more is downloaded — embeds every frame and both "
                "prompts. For each clip the stage reports the mean similarity to its own prompt, to the other capability's prompt, "
                "and the margin between them. The baseline is meaningful because the two prompts describe different scenes.\n\n"
                "**Why not a standard video metric?** Fréchet Video Distance needs hundreds of reference clips, and VBench-style "
                "suites need many more generations than two clips that each take over 20 minutes on a T4. The notebook therefore "
                "reports sanity checks and a proxy, labelled as such.\n\n"
                "**Predict before running:** will each clip be closer to its own prompt than to the other one? Which margin do you "
                "expect to be larger, and why?"
            ),
            "code": (
                "run_stage('evaluate')\n"
                "evaluation = load_record('evaluation.json')\n"
                "for capability in evaluation['capabilities']:\n"
                "    alignment = evaluation['prompt_alignment'][capability]\n"
                "    print(capability, {{'frame_checks_passed': evaluation['frame_checks'][capability]['passed'], 'own_prompt': alignment['own_prompt_similarity'], 'swapped_prompt': alignment['swapped_prompt_similarity'], 'margin': alignment['margin'], 'frames_closer_to_own_prompt': alignment['frames_closer_to_own_prompt']}})"
            ),
            "after": (
                "**What to notice:** both `frame_checks_passed` values; for each clip, the own-prompt and swapped-prompt similarities "
                "(CLIP cosine similarities for natural images and short captions typically sit between about 0.15 and 0.35) and a "
                "margin whose sign is the result; `frames_closer_to_own_prompt` near 1.0 when the margin is consistent across the "
                "clip; and the image-to-video first-frame PSNR. **Cross-capability interpretation:** the two capabilities share every "
                "component except the transformer, so a difference in their margins reflects the prompts, the conditioning image and "
                "the transformers together — one clip each cannot separate those causes.\n\n"
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "Expect positive margins: the boat prompt and the lava-field prompt describe different scenes. The image-to-video "
                "margin is often the larger one, because its frames stay close to a photograph whose content (dark rocks, palms, a "
                "coastline) overlaps its own prompt and not the boat prompt. A negative margin would be a real result to report, not "
                "a reason to re-run until it changes. Either way, a margin is CLIP's opinion of one clip, not a measure of quality.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 9. Export and provenance · [Engineering]\n\n"
                "The `report` stage writes `outputs/kandinsky5_video_provenance.json`, the record that makes a clip traceable: the "
                "notebook source revision; both model ids and revisions and the SHA-256 of each manifest; the component licences; the "
                "quantization and precision; the prompts and the image source with its digest and crop; the settings and seed; the "
                "runtime versions and GPU; the timings and memory peaks; each mp4's size and SHA-256; and the evaluation summary. It "
                "contains no credential. It then lists every file under `outputs/`."
            ),
            "code": "run_stage('report')",
            "after": (
                "**What to notice:** two mp4 files of a few MB each, two strips, the conditioning image, one JSON record per stage and "
                "the provenance record. The clips are generated, not reloaded artifacts: there is no trained artifact to reload in "
                "this notebook, so no fresh-reload check applies."
            ),
        },
        {
            "md": (
                "## 10. Optional activity: change one thing — the seed · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** The seed fixes the starting noise. Keep the prompt, steps "
                "and guidance, change only the seed from 42 to 7, and regenerate the text-to-video clip.\n\n"
                "1. **Predict:** will the new clip show the same boat and beach with small differences, or a different composition?\n"
                "2. **Change one thing:** set `RUN_ACTIVITY = True` (about 23 more minutes on a T4).\n"
                "3. **Run** the cell. 4. **Observe** the strip and `mean_abs_pixel_difference_between_seeds`. 5. **Explain** what the "
                "seed controls and what it means for a claim made from one clip.\n\n"
                "The activity is off by default and never part of the default path."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity')\n"
                "    show_image('kandinsky5_video_t2v_seed7_strip.jpg', 'text-to-video, seed 7')\n"
                "else:\n"
                "    print('Activity skipped (RUN_ACTIVITY is False).')"
            ),
            "after": (
                "<details>\n<summary>Check your reasoning</summary>\n\n"
                "A different seed usually gives a different composition — boat position, framing, light — that still follows the "
                "prompt, and the mean pixel difference between the two clips is large compared with the frame-to-frame change inside "
                "either clip. A single clip is one sample from the model's distribution for that prompt, so any conclusion from it "
                "is a statement about one sample.\n\n</details>"
            ),
        },
        {
            "md": (
                "## 11. Bring Your Own Data (optional) · [Evaluation practice]\n\n"
                "Run your own prompts and image through the same validation, encoding, generation, checks and export. **Contract:** "
                "prompts of 3..1000 characters; an image file that is JPEG, PNG or WEBP, at most 25 MiB, shorter side at least 256 px, "
                "longer side at most 8192 px, width/height between 0.5 and 2.5. The image is resized to cover 768 × 512 and "
                "centre-cropped, and the crop is reported. An input that breaks the contract stops in the `prepare` stage with a "
                "message naming the rule, before any model loads.\n\n"
                "**Where your data goes.** Your prompts and image stay inside this runtime: nothing is sent to any service, and the "
                "outputs are written under `outputs/` with a `byod_` prefix. Do not use confidential, personal or regulated images "
                "unless you are authorized to process them in this environment, and do not generate depictions of real people.\n\n"
                "**How.** On Kaggle, add your image as a dataset and set `BYOD_IMAGE_PATH` to its path under `/kaggle/input/` (there is "
                "no upload dialog). Leave a prompt empty to reuse the default one. `BYOD_CAPABILITIES` chooses `i2v`, `t2v` or `both`; "
                "each capability takes more than 20 minutes."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_T2V_PROMPT = ''  # @param {{type:\"string\"}}\n"
                "BYOD_I2V_PROMPT = ''  # @param {{type:\"string\"}}\n"
                "BYOD_CAPABILITIES = 'i2v'  # @param [\"i2v\", \"t2v\", \"both\"]\n"
                "if USE_BYOD:\n"
                "    capabilities = ['t2v', 'i2v'] if BYOD_CAPABILITIES == 'both' else [BYOD_CAPABILITIES]\n"
                "    if 'i2v' in capabilities and not BYOD_IMAGE_PATH:\n"
                "        raise ValueError('Set BYOD_IMAGE_PATH to an image file inside this runtime (for example under /kaggle/input/) for image-to-video.')\n"
                "    options = ['--tag', 'byod', '--capabilities', ','.join(capabilities)]\n"
                "    prepare = list(options)\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        prepare += ['--image', BYOD_IMAGE_PATH]\n"
                "    if BYOD_T2V_PROMPT:\n"
                "        prepare += ['--t2v-prompt', BYOD_T2V_PROMPT]\n"
                "    if BYOD_I2V_PROMPT:\n"
                "        prepare += ['--i2v-prompt', BYOD_I2V_PROMPT]\n"
                "    run_stage('prepare', *prepare)\n"
                "    run_stage('encode', *options)\n"
                "    for capability in capabilities:\n"
                "        run_stage(capability, *options)\n"
                "        show_image(f'byod_kandinsky5_video_{{capability}}_strip.jpg', f'BYOD {{capability}}')\n"
                "    run_stage('evaluate', *options)\n"
                "    run_stage('report', *options)\n"
                "else:\n"
                "    print('BYOD skipped (USE_BYOD is False).')"
            ),
        },
    ],
    "closing": [
        (
            "## Interpretation and limits\n\n"
            "Successful execution proves that the recorded repository revision, run in the documented Kaggle T4 environment, "
            "staged two digest-verified snapshots without the repository being cloned, encoded three prompts with a 4-bit encoder, "
            "generated one text-to-video and one image-to-video clip in float16, and wrote decoded-frame checks, a CLIP alignment "
            "proxy and a provenance record. It does **not** establish benchmark superiority, video quality, prompt faithfulness "
            "beyond CLIP's coarse opinion, safety of the generated content, or behaviour on other prompts, images, resolutions, "
            "lengths or GPUs. Results are tutorial evidence from one clip per capability.\n\n"
            "**Sources of variability.** Non-deterministic GPU kernels, the 4-bit quantization of the encoder, float16 rounding and "
            "the library builds can change pixels for the same seed; the seed activity in Section 10 shows how much the starting "
            "noise alone changes a clip. **Reproducibility is limited to what the notebook controls:** pinned revisions, digests, "
            "the lock, the seed and the settings.\n\n"
            "**Conclusion template.** Write your own conclusion in four sentences, each tied to a printed number: (1) the task and "
            "the two capabilities; (2) the principal result — did both clips pass the frame checks, and what were the two "
            "alignment margins against the swapped-prompt baseline? (3) the most important failure mode or uncertainty you "
            "observed (a distorted watermark, a frozen stretch, a near-zero margin, seed sensitivity); (4) what one clip per "
            "capability cannot show.\n\n"
            "**Transfer.** Apply the same checks to another use: a short clip from a photograph of a flood-prone river in your "
            "province, for example, through the BYOD branch. Before running, write down which check you expect to be weakest and "
            "why."
        ),
        (
            "## What did not work\n\n"
            "Each entry below comes from the **development probe** that sized this notebook (an engineering-only SMOKE notebook run "
            "on a Kaggle T4 on 2026-10-11 and 2026-10-12, Python 3.13 with an in-kernel install of torch 2.11.0+cu128, transformers "
            "5.16.1, diffusers 0.40.0 and bitsandbytes 0.50.2). It is development-run evidence, not evidence from this notebook's "
            "own run.\n\n"
            "1. **The distilled 16-step text-to-video checkpoint in float16 produced blank frames.** Tried: "
            "`kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers` at revision `06d0f76a`, 16 steps, guidance 1.0, "
            "float16. Observed: every decoded frame had a pixel standard deviation of 0.0. A per-step check showed the latent "
            "finite through step 11 and 48.5 % non-finite at step 12 (timestep ≈ 562): an activation overflow inside the "
            "transformer, beyond float16's range. Rejected because the T4 has no fast alternative precision (entry 2); the sft "
            "checkpoint, which needs 50 guided steps instead of 16 unguided ones, stayed finite at every step and is used here. "
            "The lesson: a faster checkpoint is not automatically usable at a lower precision, and a per-step finiteness check "
            "finds where it breaks.\n"
            "2. **bfloat16 on the T4 ran out of memory.** Tried: the same transformer in bfloat16. Observed: CUDA out of memory "
            "while requesting 41.59 GiB on a 14.56 GiB GPU. Inferred, not traced: attention fell back to a full-matrix path for "
            "this data type on this GPU. Rejected; float16 is used for every transformer and the VAE.\n"
            "3. **Free Colab cannot hold the 4-bit encoder load.** Observed: host memory peaked at 18.6–21.9 GB while the "
            "Qwen2.5-VL encoder was loaded and quantized; free Colab sessions have about 12.7 GB. Rejected; the notebook targets "
            "Kaggle (about 33.7 GB of host RAM) and refuses sessions with less than 24 GB available.\n\n"
            "Evidence: the probe logs for Kaggle kernel versions 1 and 2, kept with this repository's build record."
        ),
        (
            "## Troubleshooting\n\n"
            "- **Section 1 says the GPU is missing or too small.** Open *Settings → Accelerator* and choose *GPU T4 x2*, then *Run "
            "all* again. P100 and CPU sessions are not supported.\n"
            "- **Section 1 says host RAM is too low.** Start a fresh Kaggle session; other notebooks in the same session share memory. "
            "Free Colab cannot run this notebook.\n"
            "- **A download fails or stalls (Section 2 or 3).** Turn *Internet* on in the notebook settings and re-run the failed "
            "cell; staging fetches only files that are still missing. A `SHA-256 mismatch` names the file: delete it from the "
            "weights directory printed in Section 1 and re-run Section 3. Never edit a manifest or a lock hash.\n"
            "- **Not enough disk.** The weights need about {total_gb} GB on the temporary disk; start a fresh session.\n"
            "- **`CUDA out of memory` in Sections 5–7.** Make sure no other process uses the GPU (restart the session and *Run all*). "
            "Do not raise the resolution or frame count above the measured envelope: the validators refuse it on purpose.\n"
            "- **A stage reports non-finite or blank frames.** That is the float16 overflow failure mode; check that the pinned "
            "sft revision was staged (Section 3) and that the settings are the defaults.\n"
            "- **The sample photograph fails its digest.** The Wikimedia file has changed upstream; the notebook will not "
            "substitute another image. Use BYOD with your own image.\n"
            "- **BYOD refuses your image.** The message names the rule (format, size, side length or aspect ratio); fix the file "
            "and re-run Section 11. The Kaggle path must point to a file, not a directory."
        ),
        (
            "## References\n\n"
            "[1] V. Arkhipkin et al., \"Kandinsky 5.0: A family of foundation models for image and video generation,\" arXiv:2511.14993, 2025. [Online]. Available: https://arxiv.org/abs/2511.14993\n\n"
            "[2] Kandinsky Lab, \"Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers,\" Hugging Face model repository, revision 5c74c11b3ecfb96249b588d17aebd59f3fd9a080, 2025. [Online]. Available: https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/tree/5c74c11b3ecfb96249b588d17aebd59f3fd9a080\n\n"
            "[3] Kandinsky Lab, \"Kandinsky-5.0-I2V-Lite-5s-Diffusers,\" Hugging Face model repository, revision 8d965626ab76734922b00e359ca2d8faab2e0417, 2025. [Online]. Available: https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers/tree/8d965626ab76734922b00e359ca2d8faab2e0417\n\n"
            "[4] W. Kong et al., \"HunyuanVideo: A systematic framework for large video generative models,\" arXiv:2412.03603, 2024. [Online]. Available: https://arxiv.org/abs/2412.03603\n\n"
            "[5] S. Bai et al., \"Qwen2.5-VL technical report,\" arXiv:2502.13923, 2025. [Online]. Available: https://arxiv.org/abs/2502.13923\n\n"
            "[6] A. Radford et al., \"Learning transferable visual models from natural language supervision,\" in Proc. 38th Int. Conf. Mach. Learn. (ICML), 2021, pp. 8748–8763. [Online]. Available: https://arxiv.org/abs/2103.00020\n\n"
            "[7] P. von Platen et al., \"Diffusers: State-of-the-art diffusion models,\" GitHub repository, version 0.40.0. [Online]. Available: https://github.com/huggingface/diffusers/tree/v0.40.0\n\n"
            "[8] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, \"QLoRA: Efficient finetuning of quantized LLMs,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314\n\n"
            "[9] P. A. V. Mahusay, \"Mayon Volcano, Albay (Toe ).jpg,\" Wikimedia Commons, photograph taken 2016-08-18, CC0 1.0. [Online]. Available: https://commons.wikimedia.org/wiki/File:Mayon_Volcano,_Albay_(Toe_).jpg\n\n"
            "[10] Tencent, \"Tencent Hunyuan Community License Agreement,\" tencent/HunyuanVideo, revision 6204ad6aea1a77ff5aba337c88278bb9500eb37d. [Online]. Available: https://huggingface.co/tencent/HunyuanVideo/blob/6204ad6aea1a77ff5aba337c88278bb9500eb37d/LICENSE\n\n"
            "[11] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, \"Flow matching for generative modeling,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.02747\n\n"
            "[12] J. Ho and T. Salimans, \"Classifier-free diffusion guidance,\" arXiv:2207.12598, 2022. [Online]. Available: https://arxiv.org/abs/2207.12598\n\n"
            f"Repository model card (licences, intended use and limitations): https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md"
        ),
    ],
}
