# Development-probe logs (development-run evidence, not release evidence)

These two files are the Kaggle execution logs of an engineering-only `SMOKE` notebook that sized this repository's
tutorial. They are kept byte for byte (`.gitattributes`: `-text`) because the tutorial's **What did not work** section cites
them. They are **development-run evidence**: they record different code (an in-kernel install on Python 3.13.15) from the
tutorial notebook, and they are not execution evidence for any notebook blob in this repository (NOTEBOOK_SPEC DOC10, REL13).

| File | Kaggle kernel | Version | Run | SHA-256 |
|---|---|---|---|---|
| `kandinsky5-lite-t4-smoke_v1.log` | `kurtvalcorza/kandinsky5-lite-t4-smoke` | 1 | 2026-10-11/12, Tesla T4 | `5bae7d4f69075b023e15ebde609017876a0a9334f0456e7505996af405fa4c48` |
| `kandinsky5-lite-t4-smoke_v2.log` | `kurtvalcorza/kandinsky5-lite-t4-smoke` | 2 | 2026-10-11/12, Tesla T4 | `184e78148b9aa0a1fbdf9f73012c356f0137a75418500118f02aee5ff6831248` |

Both logs report `torch 2.11.0+cu128 cuda 12.8 | transformers 5.16.1 | diffusers 0.40.0 | … | bitsandbytes 0.50.2` on
Python 3.13.15. Each file is Kaggle's JSON stream log (a list of `{stream_name, time, data}` records).

What each log shows (search for the bracketed stage names):

- **v1:** `[load_encoders]` host-memory peak `host_rss_peak_gb` 21.94 with the 4-bit Qwen2.5-VL encoder; `[t2v_float16]` the
  distilled 16-step checkpoint in float16 returned frames with `pixel_std` 0.0 (blank); `[t2v_bfloat16]` CUDA out of
  memory requesting 41.59 GiB; `[i2v_float16]` the I2V Lite model generated 49 × 512 × 768 frames in 1378.2 s with a
  10.19 GB GPU peak.
- **v2:** `[t2v_sft_float16]` the sft text-to-video checkpoint stayed finite at every step and generated in 1352.1 s with
  a 10.19 GB GPU peak (host peak 18.55 GB); `[distilled_diag_float16]` the distilled checkpoint's latent was finite through
  step 11 and 51.5 % finite (48.5 % non-finite) at step 12, timestep ≈ 562.

The run dates are those recorded by the dispatching session (2026-10-11 and 2026-10-12); the logs carry relative times
only. The generated mp4 files of the probe are not kept here.
