# C4G motion-supervision work: handoff / server migration notes

State as of 2026-09-19. Everything below is what a fresh machine needs to reproduce the current
training runs, plus what the uncommitted work in this branch actually changes.

## 1. What is being trained right now

Two runs of the same recipe that differ only in the timestamp embedding, both from the C3G Gaussian
head init, 20k steps, davisK intrinsics, 4-dataset mix:

| run | config | wandb | state on 2026-09-19 12:00 UTC |
| --- | --- | --- | --- |
| motion v2 (target_1 / period 10000) | `config/training/c4g_motion_v2_davisK_all.yaml` | `motion_v2_davisK_all_20k` | step 18.6k of 20k, 4 GPUs |
| motion v2 (relative / period 100) | `config/training/c4g_motion_v2_relative_davisK_all.yaml` | `motion_v2_relative_p100_davisK_all_20k` | step 15.7k of 20k, 2 GPUs + accumulation 2 |

Checkpoints live under `outputs/exp_<wandb name>/<timestamp>/checkpoints/` (`last.ckpt` is the resume
point). Both jobs evaluate themselves at 20k on ADT / NVIDIA / TUM / iPhone; the target_1 job also
evaluates the July reference checkpoint on NVIDIA / TUM / iPhone for comparison.

Paused (cancelled, resumable): the C3G run `re10k_vggt518_lpf` in `../C3G` at step 69k of 100k, and a
prepared 8192-Gaussian follow-up (`re10k_vggt518_lpf_g8192`) that never started.

## 2. Environment

- Training/eval env: conda env `l40s_anysplat` (python 3.10, torch 2.5.1+cu124), created by cloning
  `anysplat`. It is self-contained: `python -m src.main ...` works on a bare node with no conda on PATH.
- The interactive sessions on this cluster ran inside the singularity sandbox `~/3d_gpt`
  (`singularity shell --nv --bind /music-3d-shared-disk/dataset:/music-3d-shared-disk/dataset/ --bind
  /music-3d-shared-disk/user/KAIST:/music-3d-shared-disk/user/KAIST ~/3d_gpt`). The image carries its own
  `/opt/conda` (python 3.11) which C3G uses. C4G does not need the image, except that `cv2` wants
  `libGL.so.1`, which the login node lacks.
- Gotcha if you run the image non-interactively: `singularity exec` does not source `~/.bashrc`, so
  `~/.local/bin` is missing from PATH, and the pip `--user` ninja wrapper in `/opt/conda/bin` then
  re-executes itself forever, hanging torch's JIT extension load. Use `bash -ic`, or prepend
  `~/.local/bin`. A JIT load killed mid-build also leaves
  `~/.cache/torch_extensions/py311_cu124/*/lock` behind, which blocks every later load.

## 3. Data

Training datasets (paths are the current cluster's; they are all in `config/dataset/*.yaml`):

| dataset | path | size | notes |
| --- | --- | --- | --- |
| kubric | `/music-3d-shared-disk/dataset` (`render_config_include: [kubric]`) | – | multi-camera; ships GT depth / flow / dynamic masks through `motion_fields` |
| spring | `/music-3d-shared-disk/dataset/Spring/` | 311G | GT depth + flow; needs the preprocessed motion maps below |
| egoexo4d_mono | `/music-3d-shared-disk/dataset/EgoExo4D/original` | large | no GT motion; the loss builds pseudo-GT online from RAFT + MoGe |
| re10k | `/music-3d-shared-disk/dataset/re10k/re10k/` | 536G | static scenes |

Derived data that has to be regenerated on a new machine:

- `scripts/preprocess_spring_motion.py` → `<preprocessing>/spring_motion_224` (4.2G): per-frame depth,
  forward flow, next-frame depth, rigidmap-derived dynamic mask and validity at 224x398 pre-crop.
  The path goes into `dataset.spring.motion_root`.
- Spring dynamic masks (`mask_root`, 958M) at
  `/music-3d-shared-disk/user/KAIST/MK/preprocessing/dynamic_masks/spring`, extracted from
  `preprocessing/dynamic_masks.tar.zst`. Note: with `motion_root` set, the photometric mask comes from the
  GT rigidmap instead, so these are only needed for runs without motion maps.

Evaluation datasets (ported into this repo, `config/evaluation/*.yaml`): ADT
(`.../InterpolatedC3G/datasets/ADT/eval/`, 7.9G), NVIDIA (`.../datasets/nvidia/nvidia/`, 69M), TUM
(`.../datasets/TUM/train/`, 879M), iPhone (`/music-3d-shared-disk/dataset/iphone/original_data`).

Weights:

- Gaussian head init: `.../c4g/C3G/pretrained_weights/gaussian_decoder.ckpt` (11G) —
  `model.encoder.pretrained_weights`.
- Downloaded automatically on first run (need network or a warm HF cache): VGGT-1B
  (`huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`), MoGe (`Ruicheng/moge-2-vitl-normal`),
  CoWTracker (`facebook/cowtracker`), RAFT (torchvision `Raft_Large_Weights.C_T_SKHT_V2`).

## 4. How to launch

On a preempting SLURM partition (this cluster's `sharedp` requeues jobs):

```bash
# train to a stop step, then evaluate that checkpoint; resumes from the newest complete checkpoint
EXTRA_EVALS=ref45k sbatch -J c4g_motion_v2 scripts/sbatch_train_eval.sh \
    c4g_motion_v2_davisK_all motion_v2_davisK_all_20k 20000 ref45k motion_v2_20k
# same on 2 GPUs (accumulation keeps the effective batch of 4) and with denser checkpoints
sbatch --gres=gpu:2 -J c4g_relative_v2 scripts/sbatch_train_eval.sh \
    c4g_motion_v2_relative_davisK_all motion_v2_relative_p100_davisK_all_20k 20000 v43 relative_v2_20k \
    trainer.accumulate_grad_batches=2 checkpointing.every_n_train_steps=250
```

Without SLURM, `scripts/train_motion_davisK.sh` (and `..._relative_...`) run the same thing directly;
add `checkpointing.load=<abs path>/last.ckpt` to resume. Evaluation alone:
`python -m src.main +evaluation={adt,nvidia,tum,iphone}_vggt_{ref45k,v43} mode=test
checkpointing.load=<ckpt> wandb.name=<name>`; pick `ref45k` configs for target_1 models and `v43` for
relative ones, because `normalize_type` is **not** stored in the checkpoint (the sinusoid period is).

Diagnostic for whether the Gaussians actually move: `scripts/diag_gaussian_motion.py` (one GPU,
`scripts/sbatch_diag_gaussian_motion.sh` as an example). It decodes fixed kubric/spring batches for
several checkpoints and reports, per checkpoint, the moving-pair displacement ratio against GT flow and
the direction cosine.

## 5. What this branch changes

Motion supervision (the main work):

- `src/loss/loss_motion.py`: 3D scene-flow, 2D flow and static-anchor losses over the per-timestamp
  Gaussians, with online RAFT+MoGe pseudo-GT for datasets without maps. v2 adds `flow3d_move` /
  `flow3d_still`: the flow residual split by whether the GT flow exceeds `flow3d_move_rel` and averaged
  separately, each normalised to O(1). It also exports `flow3d_move_ratio` / `flow3d_move_cos` as
  training-time motion health stats.
- `src/model/model_wrapper.py`: `compute_motion_loss` per batch element, weighting and wandb logging
  (`loss/motion_*`, `motion/*`), plus the printed `motion = {...}` line.
- `src/dataset/shims/motion_fields.py`, kubric `motion_fields`, spring `motion_root`, re10k
  `static_motion_mask`, and flip/crop handling of the maps.
- `src/dataset/data_module.py`: `sample_weight` per dataset and `item_weights()` so the mix is
  0.35 kubric / 0.30 egoexo / 0.25 re10k / 0.10 spring rather than uniform.
- Configs: `config/training/c4g_motion_davisK_all.yaml` (v1), `c4g_motion_v2_davisK_all.yaml` (v2) and
  the `_relative_` variants of each.

Earlier work that is also in this branch: the Omega backbone (`vggt_omega`, `omega_pose_normalizer.py`
and its configs), the ported evaluation datasets (`dataset_{adt,nvidia,tum,iphone}.py`,
`view_sampler_sequential.py`, `config/evaluation/*`), depth-supervision modes, and the token-count
ablations.

Operational scripts: `scripts/sbatch_train_eval.sh` (requeue-safe train+eval), `scripts/train_*.sh`
launchers, `scripts/preprocess_spring_motion.py`, `scripts/diag_gaussian_motion.py`.

## 6. Findings that matter for the next run

- **v1 motion losses froze the Gaussians.** With the static anchor plus a single pooled Huber flow3d
  term, moving pairs followed 13–17% of the GT flow with ~0 direction agreement, versus 58–82% for
  models trained without motion losses. Most supervision said "do not move" (the anchor covered 40–96%
  of tokens, and the pooled term is dominated by still pairs), so freezing the time dependence was the
  cheapest optimum.
- **v2 fixed the direction.** Splitting the term and normalising each part raised the kubric direction
  cosine from 0.03 (v1) to 0.69 by step 18k, with the displacement ratio climbing 0.13 → 0.19. The
  relative embedding leads the target_1 one on every dataset at matched steps.
- **`target_1` has two quirks** worth remembering: its scale depends on where the target sits in the
  window (adjacent-frame embedding distance varies ~20x), and a target equal to the first context
  timestamp divides by 1e-8 (this happens on kubric and re10k).
- **Spring's photometric mask now comes from the GT rigidmap** (intended), which makes its training loss
  ~4-5x the July reference's even though PSNR is unchanged. Compare spring on PSNR, not loss.
- Validation uses one scene per dataset, so single points swing by ±2 dB; compare windows, not points.
