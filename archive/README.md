# archive/

Parked research. None of this is in the final 0.91981 pipeline (see top-level
`README.md`). Kept for provenance — every approach here was tried and dropped.

## Dead-end methods

| File | What it tried | Why parked |
|---|---|---|
| `brainiac_*.py` (adapter/cosine/finetune/patch) | BrainIAC pretrained-encoder embeddings + adapter | Did not beat the classical PCA+Ridge map |
| `contrastive_3d_train.py` | Scratch 3D dual-encoder, contrastive, on augmented pairs | Holdout MRR ~0.04 — deep model never learned competitive embeddings |
| `slice_clip_baseline.py` | 2D slice CLIP-style features | Below classical baseline |
| `canonical_pca_retrieval.py` | Canonical PCA-axis pose normalization | Beaten by template normalization (synthetic 0.26 vs 0.55) |
| `classical_retrieval.py` | Original handcrafted feature zoo | Superseded by grid-intensity + PCA/Ridge |
| `mind_retrieval.py`, `lesion_retrieval.py` | MIND descriptors / lesion-focused features | No transfer gain |
| `siftrank_*.py` | 3D SIFT keypoint-set overlap for d2 | Promising offline, not wired into final |
| `deeds_*.py` | deedsBCV deformable-registration re-rank | Demons in `deform_lib.py` won instead |
| `d2_bspline_eval.py`, `d2_deform_submit.py`, `d2_prod_eval.py` | Earlier d2 variants | Superseded by `d2_deform_slab_submit.py` |
| `assignment_rerank.py` | Post-hoc assignment re-ranking | Reranking/fusion did NOT transfer to real Kaggle |
| `d1_error_analysis.py` | d1 failure inspection | One-off analysis |
| `diagnostic_submissions.py` | Per-dataset diagnostic CSV generator | Used during tuning only |
| `plot_lol_log.py` | Training-log plotter | Track B (dead) tooling |

## Server-only shell

`server_run_brainiac_gpu.sh`, `server_run_slice_clip_aug.sh`,
`setup_local_env_offline.sh` — remote ROCm-GPU run scripts for the dead deep-learning track.

## Superseded handoff docs

`STATUS.md`, `NEXT_AGENT.md`, `ONBOARD.md`, `SUM.md`, `SUBMISSION_PLAN.md`,
`LEAK_HUNT.md`, `REGISTRATION_RESIDUAL_REPORT.md` — session logs and handoff notes.
Their conclusions are folded into the top-level `README.md` and `docs/`.
