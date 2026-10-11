# Release status

Current status: **Candidate** — no recorded execution. The `TUTORIAL` notebook
`tutorials/kandinsky5_video_generation_colab.ipynb` (`MULTI-CAPABILITY`, `GUIDED`, NOTEBOOK_SPEC 2.4) has passed only static
and CPU checks (generator parity, validators, lock-coverage audit, unit tests). It has not been run on a GPU. The GPU recipe it
implements was sized by an engineering-only development probe on a Kaggle T4, which is not release evidence for this notebook.

Promotion needs a recorded clean-runtime `Run all` on a Kaggle T4 for the exact notebook blob, plus the BYOD journey
(NOTEBOOK_SPEC REL1–REL14), recorded in `docs/release-verification.md`.
