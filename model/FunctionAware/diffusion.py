from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import trimesh
import wandb
from tqdm import tqdm

import utils.mesh as MeshUtils
from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log
from model.Diffusion.diffusion import DiffusionNet
from model.Diffusion.diffusion_wapper import DiffusionModel
from model.Diffusion.mini_encoders import TextConditionEncoder, ZConditionEncoder
from model.Diffusion.utils.helpers import ResnetBlockFC

from .functions import FUNCTION_VOCAB
from .sdf import FunctionAwareSDFAutoEncoder


class FunctionAwareDiffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.diff_config = config["diffusion_model_paramerter"]

        diffusion_core = DiffusionNet(**self.diff_config["diffusion_model_config"])
        self.model = DiffusionModel(diffusion_core, config)
        self.text_mini_encoder = TextConditionEncoder(config)
        self.z_mini_encoder = ZConditionEncoder(config)

        fa_config = config.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.num_functions = len(self.function_vocab)
        embed_dim = int(fa_config.get("embedding_dim", 64))
        text_hat_dim = self.diff_config["diffusion_model_config"]["text_hat_dim"]
        z_hat_dim = self.diff_config["diffusion_model_config"].get("z_hat_dim") or self.diff_config.get("z_hat_dim")
        self.function_embedding = nn.Embedding(self.num_functions, embed_dim)
        self.function_to_text = nn.Sequential(
            nn.Linear(embed_dim, text_hat_dim),
            ResnetBlockFC(text_hat_dim),
        )
        self.function_to_z = nn.Linear(embed_dim, z_hat_dim)

        self.e_config = config["evaluation"]
        self.e_config["eval_mesh_output_path"] = Path(self.e_config["eval_mesh_output_path"])
        self.e_config["eval_mesh_output_path"].mkdir(parents=True, exist_ok=True)
        self.sdf = FunctionAwareSDFAutoEncoder.load_from_checkpoint(self.e_config["sdf_model_path"])
        self.sdf.eval()

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.model.parameters())
            + list(self.text_mini_encoder.parameters())
            + list(self.z_mini_encoder.parameters())
            + list(self.function_embedding.parameters())
            + list(self.function_to_text.parameters())
            + list(self.function_to_z.parameters()),
            lr=self.config["lr"],
        )

    def _condition(self, text, z, function_id):
        function_id = function_id.to(device=z.device, dtype=torch.long)
        func_emb = self.function_embedding(function_id)

        max_tau = 0.2 + self.current_epoch * self.config["tau_ratio_on_epoch"]
        min_tau = 0.19
        tau = random.uniform(min_tau, max_tau)

        z_conditions, z_KL, z_perplexity, z_logits = self.z_mini_encoder(z, tau)
        z_conditions = z_conditions + self.function_to_z(func_emb).unsqueeze(1)

        vq_loss, text_hat = self.text_mini_encoder(text)
        text_hat = text_hat + self.function_to_text(func_emb)
        return z_conditions, z_KL, z_perplexity, z_logits, vq_loss, text_hat, tau, max_tau, min_tau

    def step(self, batch, batch_idx):
        text, z, function_id = batch
        outputs = self._condition(text, z, function_id)
        z_conditions, z_KL, z_perplexity, z_logits, vq_loss, text_hat, tau, max_tau, min_tau = outputs

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_latent_1, _ = self.model.diffusion_model_from_latent(
            z,
            cond={
                "z_hat": z_conditions,
                "text": text_hat,
            },
        )

        z_KL = self.config["z_KL_ratio"] * z_KL
        loss = vq_loss + diff_loss_1 + z_KL
        return {
            "z": z,
            "function_id": function_id,
            "pred_latent_1": pred_latent_1,
            "loss": loss,
            "vq_loss": vq_loss,
            "diff_loss_1": diff_loss_1,
            "diff_100_loss_1": diff_100_loss_1,
            "diff_1000_loss_1": diff_1000_loss_1,
            "z_KL": z_KL,
            "z_perplexity": z_perplexity,
            "z_logits": z_logits,
            "tau": tau,
            "tau_max": max_tau,
            "tau_min": min_tau,
        }

    def training_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        del result["pred_latent_1"]
        del result["z"]
        del result["z_logits"]
        del result["function_id"]
        self.log_dict(result)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        if batch_idx != 0:
            return
        result = self.step(batch, batch_idx)
        if self.global_rank != 0:
            return result["loss"]

        function_id = result["function_id"]
        images = []
        for z in [result["pred_latent_1"], result["z"]]:
            batched_recon_latent = self.sdf.decode_latent(z, function_id)
            evaluation_count = min(self.e_config["count"], batched_recon_latent.shape[0], z.shape[0])
            screenshots = [np.random.randint(0, 255, (768, 1024, 3)) for _ in range(evaluation_count)]

            for batch_id in tqdm(range(evaluation_count), desc=f"Generating Mesh for Epoch = {batch_idx}"):
                output_mesh = self.e_config["eval_mesh_output_path"] / f"mesh_{self.trainer.current_epoch}_{batch_id}.ply"
                try:
                    MeshUtils.create_mesh(
                        self.sdf,
                        batched_recon_latent[[batch_id]],
                        output_mesh.as_posix(),
                        N=self.e_config["resolution"],
                        max_batch=self.e_config["max_batch"],
                        from_plane_features=True,
                        function_id=function_id[[batch_id]],
                    )
                    mesh = trimesh.load(output_mesh)
                    screenshots[batch_id] = MeshUtils.generate_mesh_screenshot(mesh)
                except Exception as e:
                    Log.error(f"Error while generating mesh: {e}")
                    continue
            images.append(np.concatenate(screenshots, axis=1))

        if getattr(self, "logger", None) is not None and hasattr(self.logger, "log_image"):
            try:
                self.logger.log_image(key="Image", images=[wandb.Image(np.concatenate(images, axis=0))])
            except Exception as e:
                Log.error(f"Error while logging images: {e}")
        return result["loss"]

