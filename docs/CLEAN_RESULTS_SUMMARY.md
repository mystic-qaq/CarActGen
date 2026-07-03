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
evaluator. Do not compare each variant's training `val_loss` directly, because
some ablations remove terms from the optimized objective.

| system | surface MAE | uniform MAE | uniform sign | plane L1 | latent std | abs(z)>1 | abs(z)>2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full function-aware | 0.01318 | 0.26022 | 0.895 | 0.01587 | 0.285 | 0.0146 | 0.0007 |
| original VAE | 0.01230 | 0.25896 | 0.902 | 0.77487 | 0.275 | 0.0155 | 0.0011 |
| no adaptive sampling | 0.01228 | 0.25890 | 0.903 | 0.02067 | 0.315 | 0.0214 | 0.0022 |
| no decoder FiLM | 0.01236 | 0.25949 | 0.902 | 0.00998 | 0.304 | 0.0196 | 0.0016 |
| no eikonal | 0.01237 | 0.25951 | 0.900 | 0.01244 | 0.305 | 0.0191 | 0.0012 |
| no FiLM conditioning | 0.01240 | 0.25942 | 0.901 | 0.01028 | 0.302 | 0.0190 | 0.0012 |
| no function loss weight | 0.01226 | 0.25934 | 0.902 | 0.01177 | 0.280 | 0.0145 | 0.0005 |
| no plane recon | 0.01278 | 0.26001 | 0.896 | 0.32179 | 0.283 | 0.0156 | 0.0007 |

Interpretation: SDF reconstruction is close across variants, so this ablation
should be discussed as a representation and functional-consistency study rather
than a pure reconstruction win. Removing plane reconstruction strongly damages
the conditioned plane representation, and removing adaptive sampling increases
latent outliers while also worsening plane consistency.

## Diffusion Generated-vs-Source Geometry

Strict articulation validity was zero for both clean pipelines, so it should not
be the main table. The fair replacement is generated-vs-source geometry.

| method | condition | CFG | group | Chamfer L1 | F@1% | F@2% |
|---|---|---:|---|---:|---:|---:|
| Original ArtFormer | zero | 0.0 | all | 0.0610 | 0.057 | 0.293 |
| Original ArtFormer | class mean | 0.0 | all | 0.0609 | 0.057 | 0.295 |
| PartLocal | text | 1.2 | all | 0.0611 | 0.064 | 0.305 |
| PartLocal | image | 1.2 | all | 0.0603 | 0.072 | 0.319 |
| PartLocal | text+image | 1.0 | all | 0.0601 | 0.072 | 0.322 |
| PartLocal | text+image | 1.2 | all | 0.0602 | 0.072 | 0.321 |

Body-only comparison:

| method | condition | CFG | Chamfer L1 | F@1% | F@2% |
|---|---|---:|---:|---:|---:|
| Original ArtFormer | zero | 0.0 | 0.0358 | 0.151 | 0.642 |
| Original ArtFormer | class mean | 0.0 | 0.0355 | 0.152 | 0.649 |
| PartLocal | text | 1.2 | 0.0321 | 0.194 | 0.741 |
| PartLocal | image | 1.2 | 0.0298 | 0.230 | 0.800 |
| PartLocal | text+image | 1.0 | 0.0297 | 0.232 | 0.802 |
| PartLocal | text+image | 1.2 | 0.0297 | 0.230 | 0.801 |

Wheel-only results remain close across methods, which supports the paper's
claim that wheel topology and assembly remain the bottleneck.

## Learned Wheel Anchors

| method | pivot L2 mean | wheelbase err | front track err | rear track err |
|---|---:|---:|---:|---:|
| train-mean template | 0.06405 | 0.07099 | 0.02062 | 0.02077 |
| BBox MLP | 0.06394 | about 0.071 | about 0.021 | about 0.021 |
| PointNet anchor | 0.03114 | 0.02850 | 0.00504 | 0.00533 |

Use the PointNet anchor predictor in qualitative figures as the learned
assembly module. The fixed template can be shown as a baseline in the viewer.

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
