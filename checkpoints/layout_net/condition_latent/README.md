# LayoutNet Checkpoint

`best.pt` is the clean-split LayoutNet checkpoint for the fixed five-part car
schema:

```text
body_shell + wheel_front_left + wheel_front_right + wheel_rear_left + wheel_rear_right
```

The network predicts body and wheel bounding boxes plus four wheel joint
anchors from clean VAE part latents and multimodal condition embeddings. It is
used by `experiments/sample_adaptive_object_multimodal_diffusion.py` when
`--layout_checkpoint` or `CARACTGEN_LAYOUT_CKPT` is set.

Clean held-out test metrics:

| method | bbox center L2 | wheel center L2 | bbox size MAE | pivot L2 | bbox IoU | wheel IoU |
|---|---:|---:|---:|---:|---:|---:|
| train-mean layout | 0.07998 | 0.08228 | 0.01997 | 0.08241 | 0.66336 | 0.61045 |
| canonical layout | 0.09139 | 0.09324 | 0.02362 | 0.09474 | 0.64720 | 0.59538 |
| LayoutNet | 0.01795 | 0.01748 | 0.00501 | 0.01751 | 0.92631 | 0.91594 |

SHA256:

```text
82e148f563691447af306795af82fc150c01a498c82e5061fb67dc3d2c303c2e  best.pt
```

