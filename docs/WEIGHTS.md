# Weights

The notebook stages two Hugging Face snapshots at immutable revisions and verifies every file's size and SHA-256 against
`weights/<key>/dimer-base-manifest.json` before any model loads. Nothing is committed except the manifests.

| Key | Repository | Revision | Files | Bytes |
|---|---|---|---|---|
| `kandinsky5-t2v-lite-sft-5s` | `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` | `5c74c11b3ecfb96249b588d17aebd59f3fd9a080` | 28 | 23,869,293,940 |
| `kandinsky5-i2v-lite-5s` | `kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers` | `8d965626ab76734922b00e359ca2d8faab2e0417` | 4 | 4,573,131,690 |

**How the manifests were built (2026-10-12).** Large files: size and SHA-256 from the Hub API
(`/api/models/<repo>/revision/<sha>?blobs=true`, LFS `sha256`). Small files: downloaded at the pinned revision, checked
against their git blob SHA-1 from the same API, then hashed with SHA-256. No weight file was downloaded to build them.

**What is excluded.** `README.md`, `.gitattributes`, `assets/`, `.ipynb_checkpoints/`, and the CLIP duplicates
`text_encoder_2/pytorch_model.bin`, `flax_model.msgpack` and `tf_model.h5` (pickle-capable or other-framework copies of
`text_encoder_2/model.safetensors`).

**Shared components.** The I2V repository's `text_encoder/`, `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`, `vae/`
files (24 paths) have the same size and SHA-256 as the T2V repository's at the pinned revisions, so only the I2V transformer,
its scheduler config and `model_index.json` are staged from it; the manifest records the shared paths under
`sharedComponentsFrom`.

**Size.** The weights set is about 28.4 GB, above the fleet's 9 GB publication convenience gate. Provenance is recorded here
and in `MODEL_CARD.md`. The gate is **open and not waived**: the owner did not approve a waiver (2026-10-12).
