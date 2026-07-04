# Paper Qualitative Viewer

This directory contains the multi-sample qualitative viewer used for the paper
comparison figures.

The viewer includes:

- `viewer.html`: the browser UI for comparing methods and assembly modes.
- `test_metadata.json`: anonymized `Test1`, `Test2`, ... labels and text
  conditions.
- `learned_anchors.json`: source/template/BBox-MLP/PointNet wheel-anchor data.
- Optional per-sample `layout_prediction.json` files: LayoutNet full-layout
  predictions written beside generated sample meshes.
- `sketches/`: the held-out DrivAerML-style sketch images shown in the side
  panel.

The generated meshes are not committed. To use this viewer after reproducing
the paper experiments, place or copy this directory under the same parent that
contains:

```text
fair_original_artformer_full/
fair_function_aware_partlocal/
```

Then serve that parent directory, for example:

```bash
python -m http.server 8031 -b 127.0.0.1 -d /path/to/paper_experiments
```

Open:

```text
http://localhost:8031/paper_qualitative/viewer.html
```

The side-panel sketch is the actual held-out sketch condition, not the rendered
generated output preview.

To enable the default `LayoutNet full layout` assembly mode for reproduced
PartLocal samples, run:

```bash
python experiments/paper_apply_layout_net_to_samples.py \
  --samples_root /path/to/paper_experiments/fair_function_aware_partlocal/eval_clean_partlocal/samples \
  --condition_root "$CARACTGEN_OUTPUT_ROOT/caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2" \
  --layout_checkpoint "$CARACTGEN_LAYOUT_CKPT"
```

If a sample does not contain `layout_prediction.json`, the viewer still works
for source/template/BBox-MLP/PointNet anchor-only comparison and reports that
the LayoutNet file is unavailable.
