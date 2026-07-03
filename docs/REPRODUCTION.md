# Reproducing CarActGen Experiments

This document describes the clean train-only protocol used for the CarActGen
paper experiments. Paths are configured through environment variables so that
scripts do not depend on a local workstation layout.

## Required Paths

```bash
export CARACTGEN_DATA_ROOT=/path/to/ArtFormer_datasets
export CARACTGEN_OUTPUT_ROOT=/path/to/caractgen_outputs
export CARACTGEN_SPLIT_PATH=$PWD/data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json
```

The dataset root is expected to contain:

```text
1_preprocessed_info/
1_preprocessed_mesh/
2_gensdf_dataset_adaptive/
2.1_text_n_latentcode/
6_encoded_drivaer_sketch_image_dinov2/
```

The split JSON must contain `train`, `val`, and `test` shape-id lists. The
test list is used only for final evaluation.

## Function-Aware VAE And PartLocal Diffusion

Set one clean train-only initializer for the function-aware VAE:

```bash
export CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT=/path/to/original_train_only_vae.ckpt
# Optional, used if available:
export CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT=/path/to/function_aware_train_only_vae.ckpt
```

Then run the monitored clean pipeline:

```bash
bash experiments/paper_run_clean_partlocal_pipeline.sh
```

Useful controls:

```bash
POLL_SEC=1800              # monitor interval
VAE_GPUS=0,1 VAE_DEVICES=2
DIFF_GPUS=2,3 DIFF_DEVICES=2
SKIP_VAE=1                # reuse an existing clean VAE checkpoint under ROOT
SKIP_DIFF=1               # reuse an existing clean PartLocal checkpoint under ROOT
SKIP_EVAL=1               # train only
```

The pipeline writes configs, manifests, logs, checkpoints, extracted latents,
and evaluation outputs under:

```text
$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/
```

## Expected Runtime

Observed runtime on our server with 8x NVIDIA RTX 3090 24GB GPUs, 256 CPU
threads, and about 247GiB RAM:

| stage | GPUs used | observed time |
|---|---:|---:|
| original train-only SDF VAE, 320 epochs | 4 | about 2 h 50 min |
| original train-only diffusion, about 2900 epochs | 4 | about 11 h 40 min |
| clean function-aware VAE continuation, 160 epochs | 8 | about 1 h |
| latent extraction with train-only statistics | 1 | about 2 min |
| clean PartLocal diffusion with monitored stopping | 4 | about 1 h 40 min |
| test-set sampling and geometry summary | 4 | about 30 min |

The default monitor polls every `POLL_SEC=1200` seconds and stops training when
the validation metric has not improved for the configured patience window. On a
smaller GPU setup, expect wall-clock time to scale roughly with the number and
memory bandwidth of available GPUs.

## VAE SDF And Latent Evaluation

```bash
python experiments/paper_vae_sdf_latent_eval.py \
  --original_ckpt /path/to/original_train_only_vae.ckpt \
  --function_ckpt /path/to/function_aware_train_only_vae.ckpt \
  --split_path "$CARACTGEN_SPLIT_PATH" \
  --eval_sdf_dataset "$CARACTGEN_DATA_ROOT/2_gensdf_dataset_adaptive" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/vae_sdf_latent"
```

## Diffusion Geometry Evaluation

```bash
python experiments/paper_diffusion_geometry_eval.py \
  --original_sample_csv /path/to/original_combined_sample_metrics.csv \
  --adaptive_sample_csv /path/to/partlocal_combined_sample_metrics.csv \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/diffusion_geometry"
```

The geometry evaluator compares each generated part mesh with the matching
held-out source part after per-mesh normalization. It reports Chamfer L1 and
F-score at 1 percent and 2 percent thresholds.

## Learned Wheel-Anchor Assembly

```bash
bash experiments/paper_run_wheel_anchor_predictors.sh
```

This trains the fixed baseline and two learned predictors:

- `train_mean_template`: fixed train-split mean anchors.
- `bbox_mlp`: predicts anchors from body bounding-box features.
- `pointnet_anchor`: predicts anchors from body point cloud plus box features.

Use `pointnet_anchor` as the learned assembly module in qualitative results.

## Qualitative Viewer

For a single generated sample, use the reusable template in
[`viewer/sample_viewer.html`](../viewer/sample_viewer.html):

```bash
export CARACTGEN_VIEWER_TEMPLATE=$PWD/viewer/sample_viewer.html

python experiments/sample_adaptive_object_multimodal_diffusion.py \
  --checkpoint "$CARACTGEN_PARTLOCAL_DIFFUSION_CKPT" \
  --dataset_path "$CARACTGEN_DATA_ROOT/2.1_text_n_latentcode" \
  --shape_id car_drivaer_305 \
  --condition_mode text_image \
  --image_embedding_dir "$CARACTGEN_DATA_ROOT/6_encoded_drivaer_sketch_image_dinov2" \
  --guidance_scale 1.2

SAMPLE_DIR=$(find "$CARACTGEN_OUTPUT_ROOT/samples" -maxdepth 1 -type d | sort | tail -1)
python -m http.server --directory "$SAMPLE_DIR" 8000
```

Then open `http://localhost:8000/viewer.html`. If the run is on a remote
server, forward the port with `ssh -L 8000:localhost:8000 user@server`.

The paper comparison viewer is generated outside git from lightweight JSON
metadata and generated meshes. It should show anonymized test labels, the text
condition, input sketch, generated parts, template assembly, and learned
PointNet anchor assembly. Generated viewer assets should not be committed to
the release branch.
