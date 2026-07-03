# Checkpoints

This directory contains small model parameter files that can be committed
directly to git. Large VAE and diffusion checkpoints used in the paper are
listed below but are not committed because they exceed normal GitHub file-size
and repository-size limits.

## Included

```text
checkpoints/wheel_anchor/pointnet_anchor/best.pt
checkpoints/wheel_anchor/bbox_mlp/best.pt
```

These are the learned wheel-anchor assembly models reported in the paper.

| model | size | SHA256 |
|---|---:|---|
| PointNet anchor | 786424 bytes | `9e7465b4751e7c70ba5f81113b1a1dc7ac07975e00f01c25c1f1edc8246936db` |
| BBox MLP anchor | 112642 bytes | `544eb777afb11f8cf30772aec35ba41ec1b8711e18cefa9205c1f2cb9c002581` |

## Large Paper Checkpoints

Upload these as GitHub Release assets, Git LFS files, or external storage
artifacts if exact checkpoint-based reproduction is required.

| artifact name | role | size | SHA256 |
|---|---|---:|---|
| `original_vae_trainonly_sdf_epoch0119_val0.00168.ckpt` | clean Original ArtFormer VAE | 840901912 bytes | `1ecbf02598d82723cc4b1cb2015984de8df5fc0f023d9c0bed9d6d4cd8cf6b69` |
| `original_diffusion_trainonly_epoch0419_val0.00795.ckpt` | clean Original ArtFormer diffusion | 1425878356 bytes | `acca130e6af843aec43f92e12a1a2c0d50920560a720337b801e1e51ed8e4f84` |
| `function_aware_vae_trainonly_epoch0139_val0.00176.ckpt` | clean function-aware VAE | 844996728 bytes | `3cdd015e944715886e991d005a88e7de453b9cc34f54293d25387af298029b4b` |
| `partlocal_diffusion_trainonly_epoch0599_val0.37401.ckpt` | clean PartLocal diffusion | 985857000 bytes | `0a5fed276407d6df59484e1ced7d93b3b1cc53a388eabeda07b6425eb0f62840` |

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
export CARACTGEN_FUNCTION_VAE_CKPT=checkpoints/large/function_aware_vae_trainonly_epoch0139_val0.00176.ckpt
```
