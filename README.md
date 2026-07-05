# CarActGen

CarActGen is a function-aware car part generation project built on top of
ArtFormer. The release branch keeps the clean train-only experimental path used
for the paper: held-out object splits, train-only latent statistics,
function-aware SDF VAE training, PartLocal multimodal diffusion, and learned
LayoutNet assembly.

## Qualitative Examples

CarActGen generates five separated car parts and assembles them with LayoutNet
into a fixed four-wheel topology. The examples below are generated from the
released clean PartLocal checkpoint and LayoutNet assembly.

| Text-conditioned generation | Sketch-conditioned generation |
|---|---|
| Prompt: [`A long vehicle with four large wheels`](resource/text.txt) | Input sketch: [`novel_drivaer_style_sketch.png`](resource/novel_drivaer_style_sketch.png) |
| <img src="resource/Text.gif" alt="Text-conditioned CarActGen generation" width="420"> | <img src="resource/sketch.gif" alt="Sketch-conditioned CarActGen generation" width="420"> |

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
- `experiments/paper_train_layout_net.py`: clean supervised LayoutNet training
  for predicting body/wheel boxes and wheel joints from part latents and
  multimodal conditions.
- `experiments/paper_apply_layout_net_to_samples.py`: postprocess saved
  PartLocal samples with LayoutNet layout predictions for the paper viewer.
- `experiments/paper_train_wheel_anchor_predictor.py`: anchor-only assembly
  ablations for separating joint placement from full layout prediction.
- `experiments/sample_adaptive_object_multimodal_diffusion.py`,
  `viewer/sample_viewer.html`, and `viewer/paper_qualitative`: single-sample
  and paper-comparison qualitative viewers.
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
- Learned LayoutNet and anchor-only ablation predictors with released small
  checkpoints.
- A reusable single-sample Three.js viewer and the paper qualitative comparison
  viewer template.
- Documentation for data, checkpoints, training, evaluation, and visualization.

Generated datasets, checkpoints, wandb runs, rendered meshes, paper scratch
outputs, and local PDFs are intentionally excluded.
Small metadata files plus LayoutNet and anchor-only checkpoints are included under
[`data/caractgen_metadata`](data/caractgen_metadata) and
[`checkpoints`](checkpoints).

## Released Data And Large Checkpoints

The full derived dataset and the four large paper checkpoints are distributed
through PKU Disk:

```text
https://disk.pku.edu.cn/anyshare/zh-cn/dir/6DAC6AE607984BBD9DE8AC53993D75FD
```

Download and extract the dataset so that `CARACTGEN_DATA_ROOT` points to the
`ArtFormer_datasets` directory. Download the large checkpoints into
`checkpoints/large/` or another local directory, then set the checkpoint
environment variables shown below.

The PKU Disk package contains the main clean checkpoints needed for the paper
pipeline: original VAE, original diffusion, function-aware VAE, and PartLocal
diffusion. VAE ablation checkpoints are not included in the uploaded package;
the ablation diagnostics can be reproduced by rerunning the ablation configs
described in [`docs/USAGE.md`](docs/USAGE.md).

Training lineage note: the released function-aware VAE is not a from-scratch
run. In the reported clean protocol, the original train-only SDF VAE is first
trained on the train split for 320 epochs, and the function-aware VAE is then
warm-started from that original VAE checkpoint with `initialize_from_sdf` and
continued for 160 epochs with validation checkpoint selection.

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
export CARACTGEN_ORIGINAL_VAE_CKPT=$PWD/checkpoints/large/original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt
export CARACTGEN_ORIGINAL_DIFFUSION_CKPT=$PWD/checkpoints/large/original_diffusion_trainonly_epoch0419_val0.00795.ckpt
export CARACTGEN_FUNCTION_VAE_CKPT=$PWD/checkpoints/large/function_aware_vae_trainonly_epoch0139_val0.00176.ckpt
export CARACTGEN_PARTLOCAL_DIFFUSION_CKPT=$PWD/checkpoints/large/partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt
export CARACTGEN_LAYOUT_CKPT=$PWD/checkpoints/layout_net/condition_latent/best.pt

# Aliases used by the clean training pipeline:
export CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT=$CARACTGEN_ORIGINAL_VAE_CKPT
export CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT=$CARACTGEN_FUNCTION_VAE_CKPT
```

`CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT` is the 320-epoch original VAE
initializer used to reproduce the paper training lineage. The
`CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT` alias points to an already trained
function-aware VAE and is mainly useful for checkpoint-based evaluation or
additional continuation. To rerun the paper function-aware VAE training from
the original 320-epoch initializer, leave `CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT`
unset and keep `CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT` set.

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

For the learned LayoutNet assembly module:

```bash
python experiments/paper_train_layout_net.py \
  --condition_root "$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2" \
  --info_root "$CARACTGEN_DATA_ROOT/1_preprocessed_info"
```

For the anchor-only ablation:

```bash
bash experiments/paper_run_wheel_anchor_predictors.sh
```

To generate one qualitative sample and open the lightweight interactive viewer:

```bash
export CARACTGEN_VIEWER_TEMPLATE=$PWD/viewer/sample_viewer.html

python experiments/sample_adaptive_object_multimodal_diffusion.py \
  --checkpoint "$CARACTGEN_PARTLOCAL_DIFFUSION_CKPT" \
  --dataset_path "$CARACTGEN_DATA_ROOT/2.1_text_n_latentcode" \
  --shape_id car_drivaer_305 \
  --condition_mode text_image \
  --image_embedding_dir "$CARACTGEN_DATA_ROOT/6_encoded_drivaer_sketch_image_dinov2" \
  --layout_checkpoint "$CARACTGEN_LAYOUT_CKPT" \
  --guidance_scale 1.2

SAMPLE_DIR=$(find "$CARACTGEN_OUTPUT_ROOT/samples" -maxdepth 1 -type d | sort | tail -1)
python -m http.server --directory "$SAMPLE_DIR" 8000
```

Then open `http://localhost:8000/viewer.html`. When running on a remote server,
forward the port with `ssh -L 8000:localhost:8000 user@server`.

The paper comparison viewer, matching the local 8031-style layout, is under
[`viewer/paper_qualitative`](viewer/paper_qualitative). It expects the reproduced
paper sample directories as siblings and uses real held-out sketch images in its
side panel. For LayoutNet assembly mode, run
`experiments/paper_apply_layout_net_to_samples.py` once on the reproduced
PartLocal sample directories so that each sample contains `layout_prediction.json`.

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

For a checkpoint-based reproduction using the released PKU Disk files, no
training is required for qualitative sampling, LayoutNet assembly, or the main
held-out geometry evaluators.

## Repository Layout

```text
configs/                  Training config examples
data/process_data_script/ Dataset conversion and feature extraction
experiments/              CarActGen training, evaluation, and paper scripts
model/FunctionAware/      Function-aware VAE and diffusion models
model/SDFAutoEncoder/     Original SDF VAE
model/Diffusion/          Original diffusion model
viewer/                   Single-sample and paper comparison viewers
docs/                     Reproduction and result notes
```

## Acknowledgement

This project extends ArtFormer, accepted to CVPR 2025:
`ArtFormer: Controllable Generation of Diverse 3D Articulated Objects`.
