# Reproducing CarActGen Experiments

This document describes the clean protocol used for the CarActGen paper
experiments. Paths are configured through environment variables so that scripts
do not depend on a local workstation layout.

## Required Paths

```bash
export CARACTGEN_DATA_ROOT=/path/to/ArtFormer_datasets
export CARACTGEN_OUTPUT_ROOT=/path/to/caractgen_outputs
export CARACTGEN_SPLIT_PATH=/path/to/splits/object_sketch_dinov2_partlocal_seed123456798.json
```

The dataset root is expected to contain:

```text
1_preprocessed_info/
1_preprocessed_mesh/
2_gensdf_dataset_adaptive/
2.1_text_n_latentcode/
6_encoded_drivaer_sketch_image_dinov2/
```

The split JSON must contain `train`, `val`, and `test` shape-id lists. The test
list must be held out from all training, checkpoint selection, and latent
normalization.

## Function-Aware VAE And PartLocal Diffusion

Set one clean initializer for the function-aware VAE:

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
$CARACTGEN_OUTPUT_ROOT/paper_experiments/fair_function_aware_partlocal/
```

## VAE SDF And Latent Evaluation

```bash
python experiments/paper_vae_sdf_latent_eval.py \
  --original_ckpt /path/to/original_train_only_vae.ckpt \
  --function_ckpt /path/to/function_aware_train_only_vae.ckpt \
  --split_path "$CARACTGEN_SPLIT_PATH" \
  --eval_sdf_dataset "$CARACTGEN_DATA_ROOT/2_gensdf_dataset_adaptive" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/paper_experiments/vae_sdf_latent"
```

## Diffusion Geometry Evaluation

```bash
python experiments/paper_diffusion_geometry_eval.py \
  --original_sample_csv /path/to/original_combined_sample_metrics.csv \
  --adaptive_sample_csv /path/to/partlocal_combined_sample_metrics.csv \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/paper_experiments/diffusion_geometry"
```

The geometry evaluator compares each generated part mesh with the matching
held-out source part after per-mesh normalization. It reports Chamfer L1 and
F-score at 1 percent and 2 percent thresholds.

## Learned Wheel-Anchor Assembly

```bash
bash experiments/paper_run_wheel_anchor_predictors.sh
```

This trains both baselines:

- `train_mean_template`: fixed train-split mean anchors.
- `bbox_mlp`: predicts anchors from body bounding-box features.
- `pointnet_anchor`: predicts anchors from body point cloud plus box features.

Use `pointnet_anchor` as the learned assembly module in qualitative results.

## Qualitative Viewer

The qualitative viewer is generated outside git with lightweight JSON metadata
and generated meshes. It should show:

- anonymized test labels such as `Test1`, `Test2`;
- the text condition;
- source/reference parts;
- generated parts;
- template, BBox MLP, and PointNet anchor assemblies.

Generated viewer assets should not be committed to the release branch. Commit
only reusable viewer-generation code or documentation.

## What Not To Claim As Main Fair Results

Routed diffusion and old all-data ablations are useful design exploration, but
they should not be mixed into main fair tables unless rerun with the same clean
split and train-only latent statistics.
