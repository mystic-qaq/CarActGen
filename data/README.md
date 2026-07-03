# Data

This repository includes a lightweight CarActGen metadata package under
`data/caractgen_metadata/`. The full geometry and SDF training data are too
large for a normal Git repository, so they are regenerated from the public
DrivAerML STL dataset using the scripts in `data/process_data_script/`.

## Source

The car geometry comes from the public DrivAerML dataset. Download the STL
meshes from the official dataset page, then arrange them as:

```text
$DRIVAERML_ROOT/
  run_1/drivaer_1.stl
  run_2/drivaer_2.stl
  ...
```

The released metadata package contains 484 usable cars. Each car is represented
as five articulated parts:

```text
body_shell
wheel_front_left
wheel_front_right
wheel_rear_left
wheel_rear_right
```

Wheel joints are inferred from the wheel component centroids, with rotation axis
`[0, 1, 0]` and a full revolution limit.

## Included In Git

```text
data/caractgen_metadata/
  1_preprocessed_info/      ArtFormer-format JSON metadata for 484 cars
  3_text_condition/         five text prompts per car
  splits/                   clean train/val/test splits
  dataset_manifest.json     counts and excluded large artifacts
```

These files are sufficient to inspect the dataset split, part hierarchy, wheel
anchors, and text conditions used in the paper.

## Regenerating Full Data

Set paths:

```bash
export DRIVAERML_ROOT=/path/to/drivaerml
export CARACTGEN_DATA_ROOT=/path/to/ArtFormer_datasets
```

Split DrivAerML STL files into 5-part car assets:

```bash
python data/process_data_script/0_split_drivaerml_stl.py \
  --stl_dataset "$DRIVAERML_ROOT" \
  --output_dir "$CARACTGEN_DATA_ROOT/0_raw_dataset/car" \
  --n_workers 32 \
  --reset_output
```

Convert the raw 5-part assets into ArtFormer preprocessed JSON and PLY files:

```bash
python data/process_data_script/1.0_extract_car_dataset.py \
  --raw_dataset_root "$CARACTGEN_DATA_ROOT/0_raw_dataset/car" \
  --output_root "$CARACTGEN_DATA_ROOT" \
  --n_process 32 \
  --reset_output
```

Generate adaptive SDF samples:

```bash
python data/process_data_script/2.1_generate_gensdf_dataset.py \
  --input_mesh_dir "$CARACTGEN_DATA_ROOT/1_preprocessed_mesh" \
  --output_dir "$CARACTGEN_DATA_ROOT/2_gensdf_dataset_adaptive" \
  --meta_json_path "$CARACTGEN_DATA_ROOT/meta.json" \
  --adaptive_sampling \
  --n_process 20
```

Text conditions in `data/caractgen_metadata/3_text_condition/` can be copied to
`$CARACTGEN_DATA_ROOT/3_text_condition/` before running the text-encoding and
latent-extraction steps described in `docs/REPRODUCTION.md`.

## Size Notes

The local full dataset used for the paper is approximately:

```text
0_raw_dataset                         14G
1_preprocessed_mesh                   6.5G
2_gensdf_dataset_adaptive             4.1G
2.1_text_n_latentcode                 161M
3_encoded_text_condition              1.2G
6_encoded_drivaer_sketch_image_dinov2 25M
```

These large derived artifacts should be distributed through GitHub Release
assets, Git LFS, or an external academic storage link rather than committed to
normal git history.
