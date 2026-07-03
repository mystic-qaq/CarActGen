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
