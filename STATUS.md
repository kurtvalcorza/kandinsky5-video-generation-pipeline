# Release status

Current status: **Candidate** — the default path of the `TUTORIAL` notebook
`tutorials/kandinsky5_video_generation.ipynb` (`MULTI-CAPABILITY`, `GUIDED`, NOTEBOOK_SPEC 2.4) passed one strict
single-pass run on a clean Kaggle T4 at commit `ae58749` / blob `659a276f` (2026-10-12, 3,198.9 s, 12/12 code cells, no
restart). The BYOD journey (REL12) has not been run. See `docs/release-verification.md`.

**Re-run pending: the notebook bytes changed after the recorded run.** The owner renamed the notebook from
`kandinsky5_video_generation_colab.ipynb` to `kandinsky5_video_generation.ipynb` on 2026-10-12 (`2a0f639`), and it was
regenerated for the new name (metadata name, carried-source hash and provenance revision changed; no code cell logic
changed). The notebook is no longer byte-identical to blob `659a276f` that ran on the Kaggle T4 at `ae58749`, so that run
does not cover the current blob. A fresh Kaggle T4 run of the current blob is pending.

**Spec deviation: KND2 (filename pattern).** NOTEBOOK_SPEC KND2 says a `TUTORIAL` notebook SHOULD be named
`<model>_<task>_colab.ipynb`. The owner chose `kandinsky5_video_generation.ipynb` (the notebook runs on Kaggle, not Colab)
and kept it after review. `tools/validate_release_assets.py` exempts this exact filename from the KND2 check and still
checks it for every other `TUTORIAL` name.

**Size gate: open, not waived.** The weights set is 28,442,425,630 bytes (28.4 GB), above the fleet's 9 GB publication convenience gate; the owner has not approved a waiver (2026-10-12).

Promotion still needs the BYOD journey on a Kaggle T4 for the exact notebook blob and an owner decision on the size gate
(NOTEBOOK_SPEC REL1–REL14), recorded in `docs/release-verification.md`.
