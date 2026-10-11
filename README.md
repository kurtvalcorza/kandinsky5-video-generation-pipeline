# kandinsky5-video-generation-pipeline

DIMER pipeline for **Kandinsky 5.0 Video Lite**: text-to-video with
`kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` (revision `5c74c11b3ecfb96249b588d17aebd59f3fd9a080`) and
image-to-video with `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers` (revision `8d965626ab76734922b00e359ca2d8faab2e0417`),
on a single Kaggle T4.

**Status: Candidate — no recorded execution.** See `STATUS.md` and `docs/release-verification.md`.

## What is here

| Path | Purpose |
|---|---|
| `tutorials/kandinsky5_video_generation_colab.ipynb` | the `TUTORIAL` notebook (`MULTI-CAPABILITY`, `GUIDED`, NOTEBOOK_SPEC 2.4), generated — do not edit by hand |
| `tutorials/requirements-kaggle.lock.txt` | hash lock for the notebook's isolated environment (Python 3.12, manylinux x86_64) |
| `src/kandinsky5_video_pipeline/` | pinned identities and the sample image (`pipeline.py`), input validation (`inputs.py`), decoded-frame checks (`checks.py`) |
| `tools/tutorial_stages.py` | the stage runner the notebook carries and runs one stage per process |
| `tools/build_notebook.py`, `tools/notebook_template.py` | the generator and the learner-facing template |
| `tools/validate_release_assets.py` | static checks of the notebook, use notice, citations and model card |
| `tools/audit_lock_coverage.py` | lock-coverage audit (import map + lazy backends) |
| `weights/*/dimer-base-manifest.json` | size and SHA-256 of every staged file, 28,442,425,630 bytes in total |
| `MODEL_CARD.md` | intended use, limitations and the licence stack (MIT transformers; Tencent Hunyuan Community License VAE) |

## Runtime

Kaggle only: a Kaggle notebook (the first cell checks `KAGGLE_KERNEL_RUN_TYPE` and `/kaggle`), GPU T4, Internet on, at least 24 GB of available host RAM. The 28.4 GB weights set is above the 9 GB publication convenience gate, which is open and not waived. Free Colab is not supported: loading the 4-bit
Qwen2.5-VL encoder peaked at 18.6–21.9 GB of host memory in development runs. The VAE licence does not apply in the EU, the UK
or South Korea.

## Development checks (CPU only)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv pytest==8.4.2 ruff==0.16.6 numpy==2.5.3 pillow==11.3.0
.venv/bin/ruff check src tests tools
.venv/bin/python -m pytest
.venv/bin/python tools/validate_release_assets.py
.venv/bin/python tools/audit_lock_coverage.py
.venv/bin/python tools/build_notebook.py --check
```

Regenerate the lock with
`uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --index-url https://pypi.org/simple --only-binary :all: -o tutorials/requirements-kaggle.lock.txt`
(`pyproject.toml` routes `torch` and `torchvision` to the explicit `pytorch-cu128` index, https://download.pytorch.org/whl/cu128, so they resolve as `+cu128` builds with CUDA 12.8 `-cu12` NVIDIA wheels from PyPI), and then the notebook with `python tools/build_notebook.py`. The notebook installs the lock with `--require-hashes --only-binary :all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`: every pin is exact (torch's `+cu128` local version exists only on the PyTorch index) and every file is hash-checked, so considering both indexes cannot substitute another build. These checks are static and are not execution evidence.

## Licence

Repository code: Apache-2.0 (`LICENSE`). Model weights keep their own licences; see `MODEL_CARD.md`.
