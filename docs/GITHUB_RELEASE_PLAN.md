# GitHub Release Plan

This repository should be released as a clean, reproducible research codebase,
not as a dump of local training state. The release branch should contain code,
configs, lightweight documentation, and scripts for reproducing the paper
pipeline. It should not contain checkpoints, generated datasets, generated
meshes, wandb runs, paper scratch files, or local absolute-path outputs.

## Recommended Release Scope

Keep:

- Core training and inference entry points:
  - `1_train_SDF.py`
  - `2_train_diff.py`
  - `3_train_trans.py`
  - `3_pred_trans.py`
- Core model code:
  - `model/SDFAutoEncoder`
  - `model/Diffusion`
  - `model/Transformer`
  - `model/FunctionAware`
- Data-processing scripts needed to rebuild the car pipeline:
  - `data/process_data_script/1.0_extract_car_dataset.py`
  - `data/process_data_script/2.1_generate_gensdf_dataset.py`
  - `data/process_data_script/2.2_generate_diff_dataset.py`
  - `data/process_data_script/2.2b_generate_diff_dataset_parallel.py`
  - `data/process_data_script/3.1_generate_car_text_condition.py`
  - `data/process_data_script/3.2_generate_encoded_text_condition.py`
  - `data/process_data_script/5_generate_text_transformer_dataset.py`
- Paper reproduction scripts:
  - `experiments/paper_dataset_stats.py`
  - `experiments/paper_vae_sdf_latent_eval.py`
  - `experiments/paper_diffusion_geometry_eval.py`
  - `experiments/paper_train_wheel_anchor_predictor.py`
  - `experiments/paper_run_clean_partlocal_pipeline.sh`
  - `experiments/paper_train_eval_full_fair_original_artformer.sh`
  - `experiments/paper_run_wheel_anchor_predictors.sh`
- Configs under `configs/`.
- Lightweight viewer source or template, without generated meshes/checkpoints.
- `env.yaml`, `sensitive_info.template.py`, and documentation.

Exclude:

- `train_root_dir/`, `wandb/`, `lightning_logs/`, `paper_experiments/`,
  local scratch-output folders, local sample folders, and all checkpoints.
- Generated SDF/latent datasets and mesh outputs.
- Local paper PDFs and scratch TeX build products.
- Any file containing absolute private paths unless documented as an example.

## Experiment Policy For The Paper

With about one day remaining, do not rerun the large VAE ablation grid or Routed
diffusion training from scratch. The clean audit already establishes the most
important story:

- strict validity is zero for both clean diffusion pipelines;
- generated-vs-source geometry is the fair quantitative replacement;
- PartLocal improves body geometry under clean text-image conditioning;
- wheel topology remains the bottleneck;
- PointNet wheel-anchor prediction is a clean, strong auxiliary contribution.

Use the old Routed work only as an architectural exploration. In the paper, keep
it in the method/discussion narrative, not in the main fair comparison tables.

## Release Branch Checklist

1. Create a fresh release branch from the current working tree after paper files
   are stable.
2. Add only source/config/docs files intentionally.
3. Run:
   - `git status --short`
   - `find . -type f -size +20M`
   - search for private absolute paths, wandb artifacts, checkpoints, and paper scratch outputs.
4. Replace absolute paths in configs with documented placeholders where possible.
5. Add a README section for:
   - dataset preprocessing;
   - function-aware VAE training;
   - PartLocal diffusion training/evaluation;
   - wheel-anchor predictor training/evaluation;
   - qualitative viewer usage.
6. Tag the release commit only after a clean clone can import the main modules.
