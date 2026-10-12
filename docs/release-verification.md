# Release verification

**Status: Candidate — default Kaggle T4 run recorded at `ae58749`; BYOD journey not yet run.**

| Date | Notebook blob / commit | Runtime | Procedure | Outcome |
|---|---|---|---|---|
| 2026-10-12 | blob `659a276f7c55fd79bc621ca001cda926581a18d8` / `ae58749337caeeb7e75f6254ba30a8589f61f16a` | Kaggle T4 clean container (kernel `kurtvalcorza/dimer-nb2-kandinsky5-video-generation` v2), Tesla T4 15,360 MiB, single GPU; kernel Python 3.13.15, isolated environment Python 3.12.12, `torch 2.11.0+cu128`, `diffusers 0.40.0`, `transformers 5.16.1`, `bitsandbytes 0.50.2`, 66 hash-locked packages installed in two single-index phases | Strict single-pass `nbclient` (`--strict-run-all`) on a fresh interpreter, notebook fetched at the commit and blob-verified, empty Hugging Face cache, no repository checkout; default path only | **PASSED** — see *Recorded executions* |

## Recorded executions

| Date | Commit / blob | Wall time | Result and evidence |
|---|---|---|---|
| 2026-10-12 | `ae58749` / `659a276f` | 3,198.9 s | **PASSED** — one pass, no restart, 0 errors, 12/12 code cells (execution counts 1–12); executed cell sources identical to the committed notebook. Encoder stage: Qwen2.5-VL 4-bit NF4 + CLIP float16, all embeddings finite, GPU peak 6.72 GB, host RSS peak 17.58 GB. T2V: 50/50 steps, no non-finite step, 1,285.1 s, GPU peak 10.19 GB; frame checks passed (49×512×768, finite, no blank or frozen frames, no abrupt cut). I2V: 50/50 steps, 1,285.5 s, GPU peak 10.20 GB; frame checks passed; first frame vs the conditioning image PSNR 33.28 dB. CLIP ViT-L/14 prompt alignment (uncalibrated tutorial evidence, swapped-prompt baseline): T2V 0.2855 vs 0.1402 (margin +0.1453), I2V 0.2940 vs 0.0998 (margin +0.1942). Videos: T2V mp4 SHA-256 `ec8385df56adb02f91dd932e3bd1efed1f27008353a1bf3a1f5b4a026c11a49a`, I2V mp4 `75a45a16d01b72fd934f059d0ad95e88e311054de1f81ad730526397c538998f` (kept with the relay record, not committed). Evidence: `docs/execution-evidence/2026-10-12/kandinsky5_video_generation_colab_ae58749_kaggle-t4.ipynb` SHA-256 `58428057395b4ea2e751f3d96070a9f01dbacf15813d049e5e9660cbc9a62598`; `docs/execution-evidence/2026-10-12/kandinsky5_video_generation_colab_ae58749_run_summary.json` SHA-256 `ceee03822cea30fbe526c8396c729ab3654beceea0cb9115a0d888dbcb6dc26e`. Not exercised: the BYOD journey (REL12) and the form widgets. One seeded clip per capability; tutorial evidence, not a benchmark |

## Coverage

| Check | Automatic (CI, CPU) | Manual (GPU) |
|---|---|---|
| Notebook is generated from the repository (`tools/build_notebook.py --check`) | yes | — |
| Carried files, lock and manifests equal the repository (parity tests) | yes | — |
| Notebook structure, guided and write-up layers, use notice, citations (`tools/validate_release_assets.py`) | yes | — |
| Lock covers runtime imports and lazy backends (`tools/audit_lock_coverage.py`) | yes (import map + recorded backends) | the library-source search needs the pinned wheels (`--library-root`) |
| Stage plumbing with stubbed models (`tests/test_stages_cpu.py`) | yes | — |
| Default `Run all` on a fresh Kaggle T4 (REL1–REL6) | no: needs a T4, about 28 GB of downloads and about an hour | recorded 2026-10-12 at `ae58749` |
| BYOD positive and negative journey (REL12) | refusals and plumbing only, with stubs | required |

CI cannot execute the notebook (GPU, 28 GB of weights, about an hour of T4 time), so every GPU figure in the repository is a
development-probe figure until a run is recorded here (REL9).

## Development probe (not release evidence)

An engineering-only `SMOKE` notebook ran on a Kaggle T4 on 2026-10-11 and 2026-10-12 (Python 3.13, in-kernel install of
torch 2.11.0+cu128, transformers 5.16.1, diffusers 0.40.0, bitsandbytes 0.50.2). It established the recipe the tutorial uses
and the three rejected approaches listed in the notebook's **What did not work** section. It ran different code from this
notebook and is not evidence for any notebook blob. Its two logs are committed byte for byte in `docs/development-probe/`.

## Kaggle runs that did not reach a result

- 2026-10-11T23:26Z, Kaggle T4, kernel `kurtvalcorza/dimer-nb2-kandinsky5-video-generation` v1, commit 89cc998: the runtime gate passed (`kaggle_run_type: Batch`); the isolated-environment install failed with a hash mismatch for `triton==3.6.0` because the mixed-index install took triton from the PyTorch mirror. Fixed by the two-phase single-index install; not an execution record for any later blob.

## Open items

- Size gate: the 28.4 GB weights set is above the 9 GB publication convenience gate; **not waived**.
- BYOD journey (REL12): not yet run on a Kaggle T4.
