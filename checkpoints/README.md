# Checkpoints

This directory contains small model parameter files that can be committed
directly to git. Large VAE and diffusion checkpoints used in the paper are
listed below but are not committed because they exceed normal GitHub file-size
and repository-size limits.

## Included

```text
checkpoints/layout_net/condition_latent/best.pt
checkpoints/wheel_anchor/pointnet_anchor/best.pt
checkpoints/wheel_anchor/bbox_mlp/best.pt
```

These are the small learned layout/assembly models reported in the paper.

| model | size | SHA256 |
|---|---:|---|
| LayoutNet condition+latent layout | 5402186 bytes | `82e148f563691447af306795af82fc150c01a498c82e5061fb67dc3d2c303c2e` |
| PointNet anchor | 786424 bytes | `9e7465b4751e7c70ba5f81113b1a1dc7ac07975e00f01c25c1f1edc8246936db` |
| BBox MLP anchor | 112642 bytes | `544eb777afb11f8cf30772aec35ba41ec1b8711e18cefa9205c1f2cb9c002581` |

## Large Paper Checkpoints

The large clean paper checkpoints are distributed through the PKU Disk release
folder:

```text
https://disk.pku.edu.cn/anyshare/zh-cn/dir/6DAC6AE607984BBD9DE8AC53993D75FD
```

Download these files into `checkpoints/large/` or another local directory and
set the environment variables below. These four checkpoints are sufficient for
checkpoint-based reproduction of the main baseline, CarActGen PartLocal
generation, LayoutNet assembly, and the main held-out geometry evaluation.

| artifact name | role | size | SHA256 |
|---|---|---:|---|
| `original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt` | clean Original ArtFormer VAE | 840901912 bytes | `1ecbf02598d82723cc4b1cb2015984de8df5fc0f023d9c0bed9d6d4cd8cf6b69` |
| `original_diffusion_trainonly_epoch0419_val0.00795.ckpt` | clean Original ArtFormer diffusion | 1425878356 bytes | `acca130e6af843aec43f92e12a1a2c0d50920560a720337b801e1e51ed8e4f84` |
| `function_aware_vae_trainonly_epoch0139_val0.00176.ckpt` | clean function-aware VAE | 844996728 bytes | `3cdd015e944715886e991d005a88e7de453b9cc34f54293d25387af298029b4b` |
| `partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt` | clean PartLocal diffusion | 985857000 bytes | `0a5fed276407d6df59484e1ced7d93b3b1cc53a388eabeda07b6425eb0f62840` |

VAE ablation checkpoints are not included in the PKU Disk package. The ablation
diagnostics in the report can be reproduced by rerunning the clean ablation
configs described in `docs/USAGE.md` and `docs/REPRODUCTION.md`. They are not
required for sampling, main metric reproduction, or viewer-based inspection.

Suggested placement after download:

```text
checkpoints/large/original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt
checkpoints/large/original_diffusion_trainonly_epoch0419_val0.00795.ckpt
checkpoints/large/function_aware_vae_trainonly_epoch0139_val0.00176.ckpt
checkpoints/large/partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt
```

Then point the reproduction scripts to these files with environment variables
or command-line arguments, for example:

```bash
export CARACTGEN_ORIGINAL_VAE_CKPT=checkpoints/large/original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt
export CARACTGEN_ORIGINAL_DIFFUSION_CKPT=checkpoints/large/original_diffusion_trainonly_epoch0419_val0.00795.ckpt
export CARACTGEN_FUNCTION_VAE_CKPT=checkpoints/large/function_aware_vae_trainonly_epoch0139_val0.00176.ckpt
export CARACTGEN_PARTLOCAL_DIFFUSION_CKPT=checkpoints/large/partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt
export CARACTGEN_LAYOUT_CKPT=checkpoints/layout_net/condition_latent/best.pt

export CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT=$CARACTGEN_ORIGINAL_VAE_CKPT
export CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT=$CARACTGEN_FUNCTION_VAE_CKPT
```
