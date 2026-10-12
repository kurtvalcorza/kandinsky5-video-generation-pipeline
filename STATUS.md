# Release status

Current status: **Candidate** — the default path of the `TUTORIAL` notebook
`tutorials/kandinsky5_video_generation_colab.ipynb` (`MULTI-CAPABILITY`, `GUIDED`, NOTEBOOK_SPEC 2.4) passed one strict
single-pass run on a clean Kaggle T4 at commit `ae58749` / blob `659a276f` (2026-10-12, 3,198.9 s, 12/12 code cells, no
restart). The BYOD journey (REL12) has not been run. See `docs/release-verification.md`.

**Size gate: open, not waived.** The weights set is 28,442,425,630 bytes (28.4 GB), above the fleet's 9 GB publication convenience gate; the owner has not approved a waiver (2026-10-12).

Promotion still needs the BYOD journey on a Kaggle T4 for the exact notebook blob and an owner decision on the size gate
(NOTEBOOK_SPEC REL1–REL14), recorded in `docs/release-verification.md`.
