# CarActGen Metadata Package

This directory contains the lightweight data files committed with the code
release.

## Contents

- `1_preprocessed_info/`: 484 ArtFormer-format JSON files. Each JSON contains
  the part list, bounding boxes, and wheel joint origins/directions.
- `3_text_condition/`: five text prompts per car used by text-conditioned
  generation.
- `splits/object_sketch_dinov2_partlocal_seed123456798.json`: the clean split
  used by the reported PartLocal experiments.
- `dataset_manifest.json`: counts and size notes.

## Clean Split

The split contains 392 train cars, 44 validation cars, and 48 test cars. The
test set is used only for final evaluation.

## Rebuilding Geometry

The mesh, SDF, latent, and sketch-feature files are not committed here. Download
the full derived `ArtFormer_datasets` package from the PKU Disk folder listed in
`data/README.md`, or regenerate the files from DrivAerML STL meshes using the
commands there.
