# Release verification

**Status: Candidate — no recorded execution.**

| Date | Notebook blob / commit | Runtime | Procedure | Outcome |
|---|---|---|---|---|

No execution of `tutorials/kandinsky5_video_generation_colab.ipynb` has been recorded.

## Coverage

| Check | Automatic (CI, CPU) | Manual (GPU) |
|---|---|---|
| Notebook is generated from the repository (`tools/build_notebook.py --check`) | yes | — |
| Carried files, lock and manifests equal the repository (parity tests) | yes | — |
| Notebook structure, guided and write-up layers, use notice, citations (`tools/validate_release_assets.py`) | yes | — |
| Lock covers runtime imports and lazy backends (`tools/audit_lock_coverage.py`) | yes (import map + recorded backends) | the library-source search needs the pinned wheels (`--library-root`) |
| Stage plumbing with stubbed models (`tests/test_stages_cpu.py`) | yes | — |
| Default `Run all` on a fresh Kaggle T4 (REL1–REL6) | no: needs a T4, about 28 GB of downloads and about an hour | required |
| BYOD positive and negative journey (REL12) | refusals and plumbing only, with stubs | required |

CI cannot execute the notebook (GPU, 28 GB of weights, about an hour of T4 time), so every GPU figure in the repository is a
development-probe figure until a run is recorded here (REL9).

## Development probe (not release evidence)

An engineering-only `SMOKE` notebook ran on a Kaggle T4 on 2026-10-11 and 2026-10-12 (Python 3.13, in-kernel install of
torch 2.11.0+cu128, transformers 5.16.1, diffusers 0.40.0, bitsandbytes 0.50.2). It established the recipe the tutorial uses
and the three rejected approaches listed in the notebook's **What did not work** section. It ran different code from this
notebook and is not evidence for any notebook blob. Its two logs are committed byte for byte in `docs/development-probe/`.

## Open items

- Size gate: the 28.4 GB weights set is above the 9 GB publication convenience gate; **not waived**.
- Lock: torch 2.11.0+cu128 / torchvision 0.26.0+cu128 from the PyTorch CUDA 12.8 index with `-cu12` NVIDIA wheels; matches the probe's torch build but has not been installed or run in the notebook's isolated environment on Kaggle.
