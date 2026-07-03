# CarActGen

CarActGen is a function-aware car part generation project built on top of
ArtFormer. The release branch keeps the clean train-only experimental path used
for the paper: held-out object splits, train-only latent statistics,
function-aware SDF VAE training, PartLocal multimodal diffusion, and learned
wheel-anchor assembly.

## Our Contributions

This repository contains a substantial amount of original ArtFormer code because
CarActGen reuses its SDF representation, training infrastructure, mesh
extraction utilities, and baseline model family. The CarActGen-specific parts
are:

- `model/FunctionAware`: function-aware SDF VAE, multimodal diffusion variants,
  part-local conditioning, and output heads for car functional parts.
- `experiments/train_function_aware_sdf.py`: clean split-aware training for the
  function-aware SDF VAE.
- `experiments/train_adaptive_object_multimodal_diffusion.py`: train/val
  split-aware PartLocal multimodal diffusion.
- `experiments/paper_run_clean_partlocal_pipeline.sh`: end-to-end clean
  training, latent extraction, monitored stopping, and held-out test evaluation.
- `experiments/paper_train_wheel_anchor_predictor.py`: learned wheel-anchor
  assembly predictors for replacing fixed template assembly.
- `experiments/sample_adaptive_object_multimodal_diffusion.py` and
  `viewer/sample_viewer.html`: sampling and interactive qualitative inspection.
- `data/caractgen_metadata`: clean object split, text conditions, source
  metadata, and manifest files needed to reproduce the reported protocol.

The original ArtFormer modules remain in `model/SDFAutoEncoder`,
`model/Diffusion`, `model/Transformer`, `eval`, and the root training scripts.
They are kept so that the repository is complete and the baseline context is
inspectable, but the recommended CarActGen reproduction path is the clean
train-only path documented below.

## What This Repository Contains

- Original ArtFormer SDF, diffusion, transformer, evaluation, and utility code.
- CarActGen function-aware VAE and PartLocal diffusion code.
- Clean train/validation/test scripts for the paper experiments.
- Learned wheel-anchor predictors and released small predictor checkpoints.
- A reusable single-sample Three.js viewer template.
- Documentation for data, checkpoints, training, evaluation, and visualization.

Generated datasets, checkpoints, wandb runs, rendered meshes, paper scratch
outputs, and local PDFs are intentionally excluded.
Small metadata files and wheel-anchor checkpoints are included under
[`data/caractgen_metadata`](data/caractgen_metadata) and
[`checkpoints/wheel_anchor`](checkpoints/wheel_anchor).

## Setup

```bash
conda env create -f env.yaml
conda activate gao

python utils/z_to_mesh/utils/libmcubes/setup.py build_ext --inplace
python utils/z_to_mesh/utils/libmise/setup.py build_ext --inplace
python utils/z_to_mesh/utils/libsimplify/setup.py build_ext --inplace
```

Set project paths with environment variables instead of editing scripts:

```bash
export CARACTGEN_DATA_ROOT=/path/to/ArtFormer_datasets
export CARACTGEN_OUTPUT_ROOT=/path/to/caractgen_outputs
export CARACTGEN_SPLIT_PATH=$PWD/data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json
```

Checkpoint variables are needed for evaluator scripts:

```bash
export CARACTGEN_ORIGINAL_VAE_CKPT=/path/to/original_train_only_vae.ckpt
export CARACTGEN_FUNCTION_VAE_CKPT=/path/to/function_aware_train_only_vae.ckpt
export CARACTGEN_PARTLOCAL_DIFFUSION_CKPT=/path/to/partlocal_diffusion_trainonly.ckpt

# Aliases used by the clean training pipeline:
export CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT=$CARACTGEN_ORIGINAL_VAE_CKPT
export CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT=$CARACTGEN_FUNCTION_VAE_CKPT
```

## Main Reproduction Pipeline

The recommended paper pipeline is documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). A shorter command-oriented
guide for training, sampling, and opening the viewer is available in
[`docs/USAGE.md`](docs/USAGE.md).

Dataset source, preprocessing, and size notes are documented in
[`data/README.md`](data/README.md). Large VAE/diffusion checkpoint checksums and
placement instructions are documented in [`checkpoints/README.md`](checkpoints/README.md).

For the main clean train-only PartLocal rerun:

```bash
bash experiments/paper_run_clean_partlocal_pipeline.sh
```

For the learned assembly ablation:

```bash
bash experiments/paper_run_wheel_anchor_predictors.sh
```

To generate one qualitative sample and open the interactive viewer:

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

Then open `http://localhost:8000/viewer.html`. When running on a remote server,
forward the port with `ssh -L 8000:localhost:8000 user@server`.

For held-out VAE and diffusion geometry metrics:

```bash
python experiments/paper_vae_sdf_latent_eval.py \
  --original_ckpt "$CARACTGEN_ORIGINAL_VAE_CKPT" \
  --function_ckpt "$CARACTGEN_FUNCTION_VAE_CKPT"

python experiments/paper_diffusion_geometry_eval.py \
  --original_sample_csv /path/to/original_combined_sample_metrics.csv \
  --adaptive_sample_csv /path/to/partlocal_combined_sample_metrics.csv
```

## Protocol

All reported model checkpoints use the train split for optimization, the
validation split for checkpoint selection, and the test split only for final
evaluation. Latent normalization statistics are computed from the train split.
The README and scripts intentionally describe this clean path only, so readers
can follow the reported protocol without internal experimental branches.

On our 8x RTX 3090 server, the clean PartLocal pipeline with an existing clean
function-aware VAE initializer took about 3 hours from VAE continuation to
test-set evaluation. A from-scratch full reproduction, including the original
baseline VAE and diffusion, should be budgeted as an overnight run.

## Repository Layout

```text
configs/                  Training config examples
data/process_data_script/ Dataset conversion and feature extraction
experiments/              CarActGen training, evaluation, and paper scripts
model/FunctionAware/      Function-aware VAE and diffusion models
model/SDFAutoEncoder/     Original SDF VAE
model/Diffusion/          Original diffusion model
viewer/                   Reusable browser viewer template
docs/                     Reproduction and result notes
```

## Acknowledgement

This project extends ArtFormer, accepted to CVPR 2025:
`ArtFormer: Controllable Generation of Diverse 3D Articulated Objects`.
