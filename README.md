# CarActGen

Function-aware car part generation built on top of ArtFormer. This repository
contains the clean train-only code path used in the CarActGen paper: a held-out
object split, function-aware SDF VAE training, PartLocal multimodal diffusion,
and the learned wheel-anchor assembly module.

## What This Repository Contains

- Original ArtFormer SDF, diffusion, and transformer code.
- Function-aware SDF VAE modules under `model/FunctionAware`.
- PartLocal object-level multimodal diffusion.
- Clean train/validation/test training and evaluation scripts for the paper.
- Wheel-anchor predictors for comparing template assembly with learned assembly.
- Documentation for reproducing the reported experiments.

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
```

## Main Reproduction Pipeline

The recommended paper pipeline is documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

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
docs/                     Reproduction and result notes
```

## Acknowledgement

This project extends ArtFormer, accepted to CVPR 2025:
`ArtFormer: Controllable Generation of Diverse 3D Articulated Objects`.
