# Release status

Current status: **Candidate** — no recorded execution. The `TUTORIAL` notebook
`tutorials/kandinsky5_video_generation_colab.ipynb` (`MULTI-CAPABILITY`, `GUIDED`, NOTEBOOK_SPEC 2.4) has passed only static
and CPU checks (generator parity, validators, lock-coverage audit, unit tests). It has not been run on a GPU. The GPU recipe it
implements was sized by an engineering-only development probe on a Kaggle T4, which is not release evidence for this notebook.

**Size gate: open, not waived.** The weights set is 28,442,425,630 bytes (28.4 GB), above the fleet's 9 GB publication convenience gate; the owner has not approved a waiver (2026-10-12).

Promotion needs a recorded clean-runtime `Run all` on a Kaggle T4 for the exact notebook blob, plus the BYOD journey
(NOTEBOOK_SPEC REL1–REL14), recorded in `docs/release-verification.md`.
