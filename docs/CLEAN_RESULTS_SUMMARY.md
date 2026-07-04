# Clean Experiment Results

These are the clean train-only results used for the current CarActGen paper
revision. The test split is used only for final evaluation.

## VAE SDF And Latent Metrics

| system | surface MAE | uniform MAE | uniform sign | plane L1 | latent std | abs(z)>1 | abs(z)>2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original VAE | 0.01230 | 0.25896 | 0.902 | 0.77487 | 0.275 | 0.0155 | 0.0011 |
| function-aware VAE | 0.01318 | 0.26022 | 0.895 | 0.01587 | 0.285 | 0.0146 | 0.0007 |

Interpretation: the function-aware VAE keeps SDF accuracy close to the original
VAE while making the conditioned plane representation much more consistent.

## Clean VAE Ablations

All variants below are trained on the clean train split, selected by validation
loss on the clean validation split, and evaluated by the same held-out test
evaluator. Do not read this as a raw reconstruction leaderboard. The clean
evidence supports a narrower conclusion: the function-aware VAE is a better
latent-to-plane interface for downstream diffusion, while raw body-dominated SDF
MAE can be worse than simpler variants.

PB is the part-balanced surface MAE over the five parts. SW is the
source-surface-area-weighted surface MAE; the body has mean weight `0.8482` on
the test split.

| system | PB surf. MAE | SW surf. MAE | plane L1 | latent std | abs(z)>1 | abs(z)>2 |
|---|---:|---:|---:|---:|---:|---:|
| original VAE | 0.01230 | 0.01075 | 0.77487 | 0.275 | 0.0155 | 0.0011 |
| full function-aware | 0.01318 | 0.01468 | 0.01587 | 0.285 | 0.0146 | 0.0007 |
| no adaptive sampling | 0.01228 | 0.01066 | 0.02067 | 0.315 | 0.0214 | 0.0022 |
| no plane recon | 0.01278 | 0.01296 | 0.32179 | 0.283 | 0.0156 | 0.0007 |

Interpretation: removing plane reconstruction strongly damages the conditioned
plane representation. Removing adaptive sampling increases latent outliers. The
other toggles were mixed in the clean benchmark and are best treated as
stabilizers rather than independent ablation wins.

## Diffusion Generated-vs-Source Geometry

Strict articulation validity was zero for both clean pipelines, so it should not
be the only table. The fair replacement is generated-vs-source geometry with
body/wheel separation and both part-balanced (PB) and source-surface-area-
weighted (SW) aggregation.

| method | condition | PB Chamfer | SW Chamfer | SW F@2% | body Chamfer | body F@2% | wheel Chamfer | wheel F@2% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original ArtFormer | zero | 0.0610 | 0.0406 | 0.576 | 0.0358 | 0.642 | 0.0674 | 0.206 |
| Original ArtFormer | class mean | 0.0609 | 0.0403 | 0.581 | 0.0355 | 0.649 | 0.0673 | 0.206 |
| PartLocal | text, cfg 1.2 | 0.0611 | 0.0376 | 0.659 | 0.0321 | 0.741 | 0.0684 | 0.196 |
| PartLocal | image, cfg 1.2 | 0.0603 | 0.0356 | 0.708 | 0.0298 | 0.800 | 0.0679 | 0.199 |
| PartLocal | text+image, cfg 1.0 | 0.0601 | 0.0355 | 0.711 | 0.0297 | 0.802 | 0.0677 | 0.202 |
| PartLocal | text+image, cfg 1.2 | 0.0602 | 0.0355 | 0.710 | 0.0297 | 0.801 | 0.0678 | 0.201 |

Wheel-only results remain close across methods, which supports the paper's
claim that wheel topology remains the bottleneck. The SW aggregate reflects that
the body contributes about 84.82 percent of held-out source surface area, so the
body improvement is visible in the visual-quality aggregate.

## Learned Wheel Anchors

| method | pivot L2 mean | wheelbase err | front track err | rear track err |
|---|---:|---:|---:|---:|
| train-mean template | 0.06405 | 0.07099 | 0.02062 | 0.02077 |
| BBox MLP | 0.06394 | about 0.071 | about 0.021 | about 0.021 |
| PointNet anchor | 0.03114 | 0.02850 | 0.00504 | 0.00533 |

Use this table as an anchor-only diagnostic ablation. PointNet shows that body
surface evidence helps joint placement, while the final assembly module is
LayoutNet because it predicts the full fixed-schema layout.

## LayoutNet Full Layout Prediction

LayoutNet predicts the fixed five-part car layout: body bbox, four wheel bboxes,
and four wheel joint anchors. It is trained only on the clean train split and
selected by validation loss.

| method | bbox center L2 | wheel center L2 | bbox size MAE | pivot L2 | bbox IoU | wheel IoU |
|---|---:|---:|---:|---:|---:|---:|
| train-mean layout | 0.07998 | 0.08228 | 0.01997 | 0.08241 | 0.66336 | 0.61045 |
| canonical layout | 0.09139 | 0.09324 | 0.02362 | 0.09474 | 0.64720 | 0.59538 |
| LayoutNet | 0.01795 | 0.01748 | 0.00501 | 0.01751 | 0.92631 | 0.91594 |

Use `checkpoints/layout_net/condition_latent/best.pt` with
`experiments/sample_adaptive_object_multimodal_diffusion.py --layout_checkpoint`
to generate `structure.json` without borrowing a held-out source bbox.
