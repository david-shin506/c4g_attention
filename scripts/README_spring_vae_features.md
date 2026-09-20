# Spring Gaussian VAE features (8 reference + 15 target frames)

The pipeline learns a feature branch on frozen C4G step-45000 geometry and exports normalized, independent-frame Wan VAE latents at 480x480 RGB resolution. It does not train the diffusion model.

Default dataset:
`/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_VAE_480x480_ctx8`

Current fixed-bounds regeneration run: `outputs/spring_vae_feature_fixed_bounds_20260913`.

```bash
/home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python -u scripts/run_spring_fixed_bounds.py
```

This controller trains from scratch with fixed `near=0.1`, `far=100` in normalized coordinates, without division by the camera baseline. Baseline pose normalization is unchanged. The `--fixed-bounds` option explicitly enables the same policy in training and both exporters; legacy loader defaults remain compatible with older experiments.

At step 1000, `training/checkpoint_step_1000.ckpt` is saved regardless of the current best validation loss. Training continues to step 4000. If the minimum validation MSE at steps 1250–4000 is strictly lower than the step-1000 MSE, training resumes (including optimizer and RNG state) to step 8000 and chooses `best.ckpt`; otherwise it chooses the exact step-1000 checkpoint. `extension_decision.json` and `selection.json` record the decision and selected checkpoint hash. Validation is the same four windows / twelve frames every 250 steps.

The selected checkpoint generates 8-context / 15-target VAE data plus **480x480 RGB predictions rendered from the same Gaussians and target cameras**. These RGB images are not VAE-decoded latent previews. The controller also regenerates the separate `Spring_480x480` RGB dataset with its existing 17-context / 33-target layout. Both exports are fully audited on completion, including fixed bounds and frame mappings. Shared clean VAE/RGB caches do not depend on near/far and are reused. Old output roots are renamed to the backup paths in `backup_manifest.json` before this controller starts.

Progress: `pipeline_status.json`, `pipeline.log`, and `training/{status.json,metrics.jsonl,validation.jsonl}`. Resume the controller using the same command after a failure; completed exports and checkpoints are preserved. The GPU allocation must remain active.

## Data contract

- Spring train, same exclusions and baseline filtering as the C4G dataset.
- 8 left-camera reference frames at offsets 0,2,...,14; 15 target frames at offsets 0,...,14; start stride 2.
- 1790 windows across 31 scenes; 3983 unique clean frames; 26850 separate predicted feature frames.
- Same central crop as the 224x224 C4G encoder, extracted from the original RGB and resized to 480x480 for the VAE.
- Independent image encoding: one time step per VAE input, no temporal compression or padding. Different images may be batched along B, never along T.
- Teacher uses the downloaded Wan2.1 VAE in BF16 compute and stores FP16 normalized latent means. `single_encode` applies Wan's `(mu-mean)/std`. These values may be negative and are not RGB-clamped or unit-normalized.
- Output shape per frame: `(16,60,60)`, saved as uncompressed FP16 `.npy` without pickle.
- Teacher input features are bilinearly resized 60x60 ->16x16 to match the 224/14 patch grid. Supervision and final feature rendering remain 60x60.
- Frozen C4G attention Q/K and x stream; trainable feature value/output projections, feature FFNs, normalization and 16-channel head, following the HG Instill structure. All original parameters remain frozen.
- Feature rasterizer is vendored separately in `submodules/diff_gaussian_rasterization_vae16` with 16 channels and a unique extension name. Rendering uses normalized DAVIS K, normalized Spring target poses and a 0.3 pixel low-pass term at 60x60.

## Training and checks

Defaults: AdamW, LR 1e-4, weight decay 0.01, gradient clipping 1.0, 1000 steps in the pilot wrapper. Each step samples a window and supervises one reference timestamp and one intermediate timestamp using latent MSE. Scenes 0014 and 0038 are excluded from feature-branch training and used for fixed validation every 250 steps. This is a feature-branch validation split; the original C4G checkpoint was previously trained on Spring.

`best.ckpt` is selected by held-out feature MSE; `latest.ckpt` includes optimizer and random-generator states for resume. Only feature weights are saved in these checkpoints, with a pointer to the frozen base checkpoint. Full export uses `best.ckpt` and records its hash and training step. No claim of equivalence to temporally compressed video latents is made; future VACE integration must use framewise encode/decode and matching latent lengths.

Verified before the full run:

- exact preservation of original Gaussian outputs on a real Spring sample;
- finite nonzero feature gradients and no gradients on original C4G parameters;
- five-step real-data learning and a two-window export/load/validation round trip;
- independent-frame VAE batching comparison;
- feature checkpoint reload, patch-count mismatch rejection, and negative feature preservation.

## Output layout

```text
clean_latents/<scene>/left/<frame>.npy
samples/<sample_id>/prediction/<frame>.npy
samples/<sample_id>/rgb_prediction/<frame>.png
samples/<sample_id>/cameras.npz
samples/<sample_id>/sample.json
clean_index.json             # source paths and unique frame IDs
plan.json                    # all windows and rejected baselines
dataset.json                 # image, VAE, geometry, index and dtype conventions
feature_checkpoint.json      # selected feature checkpoint provenance
prepare_status.json          # shared teacher cache status
status.json                  # prediction export status
index.json                   # completed prediction manifests
validation.json              # full final integrity audit
```

Manifest paths are relative to the dataset root. Context and target GT paths may point to the same shared clean latent. Prediction paths always include sample_id because the same target frame can have different input windows. `camera_path` stores both model-space camera arrays and source W2C poses. Every frame index is zero-based; original Spring filenames use index+1.

Latent payload estimate: 3.55 GB for predictions plus shared clean GT, excluding RGB predictions and training checkpoints. RGB prediction PNGs add scene-dependent storage. `pipeline_status.json`, `status.json`, `metrics.jsonl`, `validation.jsonl`, and `pipeline.log` in the training run report progress. `source_snapshot/` and `implementation.json` record the implementation used for this run.

The user selected a 1000-step pilot after the original 8000-step job had already advanced. That job was stopped at reported step 4450, before export. Its best checkpoint was exactly step 1000; an immutable copy is in `pilot_1000/checkpoint_step_1000.ckpt`, with metrics and provenance in `pilot_1000/report.json`. The later-step latest checkpoint is retained separately and is not the pilot result.
