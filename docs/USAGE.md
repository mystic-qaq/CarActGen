# CarActGen Usage Guide

This guide gives the shortest reproducible path for setting up the repository,
training the clean CarActGen models, sampling a car, and inspecting the result
with the browser viewer.

## 1. Environment

```bash
conda env create -f env.yaml
conda activate gao

python utils/z_to_mesh/utils/libmcubes/setup.py build_ext --inplace
python utils/z_to_mesh/utils/libmise/setup.py build_ext --inplace
python utils/z_to_mesh/utils/libsimplify/setup.py build_ext --inplace
```

## 2. Paths

Download the full derived dataset and the four large main checkpoints from the
PKU Disk release folder:

```text
https://disk.pku.edu.cn/anyshare/zh-cn/dir/6DAC6AE607984BBD9DE8AC53993D75FD
```

Dataset details are in [`data/README.md`](../data/README.md). Checkpoint names,
checksums, and placement instructions are in
[`checkpoints/README.md`](../checkpoints/README.md). Then set:

```bash
export CARACTGEN_DATA_ROOT=/path/to/ArtFormer_datasets
export CARACTGEN_OUTPUT_ROOT=/path/to/caractgen_outputs
export CARACTGEN_SPLIT_PATH=$PWD/data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json

export CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT=$PWD/checkpoints/large/original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt
export CARACTGEN_ORIGINAL_DIFFUSION_CKPT=$PWD/checkpoints/large/original_diffusion_trainonly_epoch0419_val0.00795.ckpt
export CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT=$PWD/checkpoints/large/function_aware_vae_trainonly_epoch0139_val0.00176.ckpt
export CARACTGEN_PARTLOCAL_DIFFUSION_CKPT=$PWD/checkpoints/large/partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt
export CARACTGEN_LAYOUT_CKPT=$PWD/checkpoints/layout_net/condition_latent/best.pt

# Aliases used by metric scripts:
export CARACTGEN_ORIGINAL_VAE_CKPT=$CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT
export CARACTGEN_FUNCTION_VAE_CKPT=$CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT
```

The split file contains 392 train cars, 44 validation cars, and 48 test cars.
Training uses only `train`, checkpoint selection uses only `val`, and final
numbers are computed only on `test`.

## 3. Train The Main CarActGen Pipeline

Run the clean PartLocal pipeline:

```bash
bash experiments/paper_run_clean_partlocal_pipeline.sh
```

The script writes a timestamped run under:

```text
$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/runs/
```

It also writes configs, checkpoints, extracted latent datasets, and evaluation
outputs under:

```text
$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/
```

Useful controls:

```bash
POLL_SEC=1800 bash experiments/paper_run_clean_partlocal_pipeline.sh

VAE_GPUS=0,1 VAE_DEVICES=2 \
DIFF_GPUS=2,3 DIFF_DEVICES=2 \
bash experiments/paper_run_clean_partlocal_pipeline.sh

SKIP_VAE=1 bash experiments/paper_run_clean_partlocal_pipeline.sh
SKIP_DIFF=1 bash experiments/paper_run_clean_partlocal_pipeline.sh
SKIP_EVAL=1 bash experiments/paper_run_clean_partlocal_pipeline.sh
```

`POLL_SEC` controls how often the monitor checks validation loss. The default
pipeline stops long runs when validation loss has stopped improving for the
configured patience window.

## 4. Train Learned Assembly

Train the fixed-schema LayoutNet, which predicts body/wheel boxes and four
wheel joint anchors from clean VAE part latents plus text/image conditions:

```bash
python experiments/paper_train_layout_net.py \
  --condition_root "$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/caractgen_layout_net"
```

The released checkpoint is already available at
`checkpoints/layout_net/condition_latent/best.pt`.

For the lighter wheel-anchor-only ablation:

```bash
bash experiments/paper_run_wheel_anchor_predictors.sh
```

This trains and evaluates:

- `train_mean_template`: fixed train-split mean anchors.
- `bbox_mlp`: learned anchors from body bounding-box features.
- `pointnet_anchor`: learned anchors from body point cloud and box features.

Use these anchor-only predictors as ablations. The main qualitative assembly
mode is LayoutNet because it predicts both part boxes and wheel anchors.

## 5. Evaluate Reported Metrics

The released PKU Disk package does not include VAE ablation checkpoints. The
main paper pipeline and qualitative viewer do not need them. Run this section
only if you want to regenerate the VAE diagnostic ablations from the clean
split.

To regenerate clean VAE ablation configs from the same original train-only VAE
initializer:

```bash
python experiments/paper_prepare_clean_vae_ablations.py \
  --output_root "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean" \
  --base_dataset "$CARACTGEN_DATA_ROOT" \
  --split_path "$CARACTGEN_SPLIT_PATH" \
  --original_trainonly_vae "$CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT"
```

Then train any generated ablation config with:

```bash
python experiments/train_function_aware_sdf.py \
  -c "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean/configs/no_decoder_film.yaml" \
  --devices 1
```

After all variants finish, evaluate them with one common held-out metric suite:

```bash
python experiments/paper_vae_ablation_sdf_latent_eval.py \
  --ablation_root "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean" \
  --original_ckpt "$CARACTGEN_ORIGINAL_VAE_CKPT" \
  --full_ckpt "$CARACTGEN_FUNCTION_VAE_CKPT" \
  --split_path "$CARACTGEN_SPLIT_PATH" \
  --eval_sdf_dataset "$CARACTGEN_DATA_ROOT/2_gensdf_dataset_adaptive" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean/eval_sdf_latent"
```

VAE reconstruction and latent metrics:

```bash
python experiments/paper_vae_sdf_latent_eval.py \
  --original_ckpt "$CARACTGEN_ORIGINAL_VAE_CKPT" \
  --function_ckpt "$CARACTGEN_FUNCTION_VAE_CKPT" \
  --split_path "$CARACTGEN_SPLIT_PATH" \
  --eval_sdf_dataset "$CARACTGEN_DATA_ROOT/2_gensdf_dataset_adaptive" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/vae_sdf_latent"
```

Diffusion geometry metrics:

```bash
python experiments/paper_diffusion_geometry_eval.py \
  --original_sample_csv /path/to/original_combined_sample_metrics.csv \
  --adaptive_sample_csv /path/to/partlocal_combined_sample_metrics.csv \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/diffusion_geometry"
```

For the source-surface-area-weighted aggregate reported in the paper, run the
geometry evaluator first and then aggregate the per-part CSV with source mesh
areas:

```bash
python experiments/paper_surface_weighted_summary.py \
  --part_metrics_csv "$CARACTGEN_OUTPUT_ROOT/diffusion_geometry/diffusion_geometry_part_metrics.csv" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/diffusion_geometry" \
  --group_cols family entry \
  --metric_names chamfer_l1 chamfer_l2 fscore_1pct fscore_2pct s2g_p95 g2s_p95
```

For VAE diagnostic tables, the same script can aggregate SDF/latent per-part
metrics:

```bash
python experiments/paper_surface_weighted_summary.py \
  --part_metrics_csv "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean/eval_sdf_latent/vae_ablation_sdf_latent_part_metrics.csv" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_OUTPUT_ROOT/vae_ablations_clean/eval_sdf_latent" \
  --group_cols system \
  --metric_names surface_mae uniform_mae surface_sign_acc uniform_sign_acc plane_l1 latent_std latent_abs_gt1 latent_abs_gt2 latent_l2
```

## 6. Generate A Qualitative Sample

Set the viewer template:

```bash
export CARACTGEN_VIEWER_TEMPLATE=$PWD/viewer/sample_viewer.html
```

Sample a held-out car from the trained PartLocal model. The sampling script
needs text/image condition embeddings, so the simplest input directory is the
base text-latent directory from the released dataset:

```bash
python experiments/sample_adaptive_object_multimodal_diffusion.py \
  --checkpoint "$CARACTGEN_PARTLOCAL_DIFFUSION_CKPT" \
  --dataset_path "$CARACTGEN_DATA_ROOT/2.1_text_n_latentcode" \
  --shape_id car_drivaer_305 \
  --condition_mode text_image \
  --image_embedding_dir "$CARACTGEN_DATA_ROOT/6_encoded_drivaer_sketch_image_dinov2" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info" \
  --mesh_root "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_root "$CARACTGEN_OUTPUT_ROOT/samples" \
  --layout_checkpoint "$CARACTGEN_LAYOUT_CKPT" \
  --guidance_scale 1.2 \
  --sdf_resolution 128
```

If you have just run `experiments/paper_run_clean_partlocal_pipeline.sh`, you
may also use the extracted clean latent dataset:

```bash
--dataset_path "$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2"
```

Valid `--condition_mode` values are `unconditional`, `text`, `image`, and
`text_image`. The sample directory contains:

```text
generated_part_mesh/   generated part PLY files
structure.json         LayoutNet boxes and joint anchors when enabled
layout_prediction.json raw LayoutNet prediction metadata when enabled
processed_nodes.pkl    ArtFormer-style structure object
pose_000.png           quick rendered preview
viewer.html            interactive browser viewer
```

For existing saved PartLocal samples, apply LayoutNet without re-decoding
meshes:

```bash
python experiments/paper_apply_layout_net_to_samples.py \
  --samples_root /path/to/paper_experiments/fair_function_aware_partlocal/eval_clean_partlocal/samples \
  --condition_root "$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2" \
  --layout_checkpoint "$CARACTGEN_LAYOUT_CKPT" \
  --overwrite
```

## 7. Open The Viewer

The single-sample viewer uses browser `fetch`, so serve the sample directory
over HTTP:

```bash
SAMPLE_DIR=$(find "$CARACTGEN_OUTPUT_ROOT/samples" -maxdepth 1 -type d | sort | tail -1)
python -m http.server --directory "$SAMPLE_DIR" 8000
```

Open:

```text
http://localhost:8000/viewer.html
```

If the server is remote:

```bash
ssh -L 8000:localhost:8000 user@server
```

Then open the same local URL in your browser. The viewer supports camera orbit,
wheel rotation, wireframe mode, and target bounding-box display.

## 8. Paper Comparison Viewer

The paper-style qualitative viewer is under
[`viewer/paper_qualitative`](../viewer/paper_qualitative). It is the
multi-sample method/assembly comparison viewer, while
[`viewer/sample_viewer.html`](../viewer/sample_viewer.html) is only a lightweight
single-sample viewer copied by the sampling script.

After reproducing the paper sample outputs, copy or symlink
`viewer/paper_qualitative` into the parent directory that also contains:

```text
fair_original_artformer_full/
fair_function_aware_partlocal/
```

Then serve that parent:

```bash
python -m http.server 8031 -b 127.0.0.1 -d /path/to/paper_experiments
```

Open:

```text
http://localhost:8031/paper_qualitative/viewer.html
```

This viewer shows real held-out sketch conditions from `sketches/`, not the
rendered generated preview. The object labels are anonymized as `Test1`,
`Test2`, ... so they are not confused with dataset names. It supports
source/template/BBox-MLP/PointNet anchor-only modes and LayoutNet full-layout
mode. To enable LayoutNet, make sure the PartLocal sample directories contain
`layout_prediction.json` files from the command above.

## 9. Expected Runtime

On our 8x RTX 3090 server, the clean PartLocal path with an existing clean
function-aware VAE initializer took about 3 hours from VAE continuation to
test-set evaluation. A full reproduction including original baseline VAE and
diffusion should be budgeted as an overnight run. More detailed runtime numbers
are in [`REPRODUCTION.md`](REPRODUCTION.md).
