# CarActGen

Function-aware car part generation built on top of ArtFormer. This repository
contains the code used for clean train/validation/test experiments, PartLocal
multimodal diffusion, and the learned wheel-anchor assembly module used in the
CarActGen paper.

## What This Repository Contains

- Original ArtFormer SDF, diffusion, and transformer code.
- Function-aware SDF VAE modules under `model/FunctionAware`.
- Object-level multimodal diffusion variants, including `partlocal_object`.
- Clean split training and evaluation scripts for the paper.
- Wheel-anchor predictors for comparing template assembly with learned assembly.
- Documentation for reproducing the fair experiments and preparing a GitHub
  release.

Generated datasets, checkpoints, wandb runs, rendered meshes, paper scratch
outputs, and local PDFs are intentionally excluded.

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
export CARACTGEN_SPLIT_PATH=/path/to/splits/object_sketch_dinov2_partlocal_seed123456798.json
```

Checkpoint variables are needed for evaluator scripts:

```bash
export CARACTGEN_ORIGINAL_VAE_CKPT=/path/to/original_train_only_vae.ckpt
export CARACTGEN_FUNCTION_VAE_CKPT=/path/to/function_aware_train_only_vae.ckpt
```

## Clean Experiment Pipeline

The recommended paper pipeline is documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

For the main clean PartLocal rerun:

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

## Fairness Policy

The paper results should use a clean object split: test shapes must be held out
from VAE training, VAE checkpoint selection, diffusion training, diffusion
checkpoint selection, and latent normalization statistics. The release scripts
preserve this policy by reading `CARACTGEN_SPLIT_PATH` and by computing latent
statistics on the train split only.

Old all-data experiments and Routed diffusion runs should be treated as
architectural exploration unless they are rerun under the same clean protocol.

## Repository Layout

```text
configs/                  Training config examples
data/process_data_script/ Dataset conversion and feature extraction
experiments/              CarActGen training, evaluation, and paper scripts
model/FunctionAware/      Function-aware VAE and diffusion models
model/SDFAutoEncoder/     Original SDF VAE
model/Diffusion/          Original diffusion model
docs/                     Reproduction and release notes
```

## Acknowledgement

This project extends ArtFormer, accepted to CVPR 2025:
`ArtFormer: Controllable Generation of Diverse 3D Articulated Objects`.
