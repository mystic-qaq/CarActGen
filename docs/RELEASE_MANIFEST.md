# Release Manifest

Use this manifest when creating the public GitHub release branch. The current
working tree contains useful code plus deadline scratch files, so do not use
`git add .`.

## Add Intentionally

Core entry points:

```bash
git add README.md .gitignore env.yaml sensitive_info.template.py
git add 1_train_SDF.py 2_train_diff.py 3_train_trans.py 3_pred_trans.py demo.py
```

Core packages:

```bash
git add model/SDFAutoEncoder model/Diffusion model/Transformer model/FunctionAware
git add utils eval static
```

CarActGen configs:

```bash
git add configs/1_SDF/train_function_aware_car_adaptive.yaml
git add configs/2_Diff/train_adaptive_object_multimodal_car_sketch_dinov2_partlocal.yaml
git add configs/2_Diff/train_adaptive_object_multimodal_car_sketch_dinov2_routed.yaml
```

Dataset scripts:

```bash
git add data/README.md data/caractgen_metadata
git add data/process_data_script/0_split_drivaerml_stl.py
git add data/process_data_script/1.0_extract_car_dataset.py
git add data/process_data_script/2.1_generate_gensdf_dataset.py
git add data/process_data_script/2.2_generate_diff_dataset.py
git add data/process_data_script/2.2b_generate_diff_dataset_parallel.py
git add data/process_data_script/3.1_generate_car_text_condition.py
git add data/process_data_script/3.2_generate_encoded_text_condition.py
git add data/process_data_script/5_generate_text_transformer_dataset.py
```

Paper reproduction scripts:

```bash
git add experiments/train_function_aware_sdf.py
git add experiments/train_adaptive_object_multimodal_diffusion.py
git add experiments/extract_function_aware_latents.py
git add experiments/sample_adaptive_object_multimodal_diffusion.py
git add experiments/fixed_car_template.py
git add experiments/paper_prepare_clean_partlocal.py
git add experiments/paper_run_clean_partlocal_pipeline.sh
git add experiments/paper_diffusion_sample_eval.py
git add experiments/paper_summarize_diffusion_samples.py
git add experiments/paper_vae_sdf_latent_eval.py
git add experiments/paper_diffusion_geometry_eval.py
git add experiments/paper_train_wheel_anchor_predictor.py
git add experiments/paper_run_wheel_anchor_predictors.sh
```

Documentation:

```bash
git add docs checkpoints/README.md
git add -f checkpoints/wheel_anchor
```

## Do Not Add

```text
CarActGen.tex
attachment/ArtFormer.pdf
wandb/
train_root_dir/
outputs/
paper_experiments/
local_scratch_outputs/
experiments/*_202607*.md
experiments/deadline_*
experiments/run_paper_experiment_queue.sh
experiments/rescue_trainonly_vae_manifest.json
*.ckpt
*.pt
*.npy
*.npz
*.ply
*.obj
*.glb
*.stl
```

## Pre-Push Checks

```bash
git status --short
find . -type f -size +20M -not -path './.git/*'
rg -n 'your-private-data-root|your-private-home-root' README.md docs configs experiments model data *.py *.sh
python3 -m py_compile \
  experiments/paper_prepare_clean_partlocal.py \
  experiments/extract_function_aware_latents.py \
  experiments/paper_train_wheel_anchor_predictor.py \
  experiments/paper_vae_sdf_latent_eval.py \
  experiments/paper_diffusion_geometry_eval.py \
  experiments/sample_adaptive_object_multimodal_diffusion.py
bash -n experiments/paper_run_clean_partlocal_pipeline.sh experiments/paper_run_wheel_anchor_predictors.sh
```
