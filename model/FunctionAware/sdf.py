from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from einops import reduce
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from tqdm import tqdm
import trimesh

from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.encoder import Encoder
from model.SDFAutoEncoder.decoder import Decoder
from model.SDFAutoEncoder.intermediate import VAE
import utils.mesh as MeshUtils
from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log

from .functions import FUNCTION_VOCAB, FUNCTION_TO_ID


class FunctionFiLMDecoder(nn.Module):
    def __init__(
        self,
        latent_size=256,
        hidden_dim=512,
        skip_connection=True,
        tanh_act=False,
        input_size=None,
        num_functions=8,
        function_embedding_dim=64,
        film_scale=0.1,
        geo_init=True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size + 3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act
        self.hidden_dim = hidden_dim
        self.film_scale = float(film_scale)

        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim
        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.block3 = nn.Linear(hidden_dim, 1)

        self.function_embedding = nn.Embedding(num_functions, function_embedding_dim)
        self.block1_film = nn.Linear(function_embedding_dim, hidden_dim * 2)
        self.block2_film = nn.Linear(function_embedding_dim, hidden_dim * 2)
        self.output_shift = nn.Embedding(num_functions, 1)
        self._init_function_adapters()

        if geo_init:
            for m in self.block3.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.000001)
                    init.constant_(m.bias, -0.5)
            for m in self.block2.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)
            for m in self.block1.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

    def _init_function_adapters(self):
        nn.init.zeros_(self.block1_film.weight)
        nn.init.zeros_(self.block1_film.bias)
        nn.init.zeros_(self.block2_film.weight)
        nn.init.zeros_(self.block2_film.bias)
        nn.init.zeros_(self.output_shift.weight)

    def _normalize_function_id(self, function_id, batch_size: int, device) -> torch.Tensor:
        if function_id is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(function_id):
            function_id = torch.tensor(function_id, dtype=torch.long, device=device)
        function_id = function_id.to(device=device, dtype=torch.long)
        if function_id.ndim == 0:
            function_id = function_id.repeat(batch_size)
        return function_id

    def _film(self, features: torch.Tensor, function_id: torch.Tensor, layer: nn.Linear):
        emb = self.function_embedding(function_id)
        gamma_beta = layer(emb).view(features.shape[0], 1, 2, self.hidden_dim)
        gamma = torch.tanh(gamma_beta[:, :, 0]) * self.film_scale
        beta = gamma_beta[:, :, 1] * self.film_scale
        return features * (1.0 + gamma) + beta

    def forward(self, x, function_id=None):
        function_id = self._normalize_function_id(function_id, x.shape[0], x.device)
        block1_out = self._film(self.block1(x), function_id, self.block1_film)

        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1)
        else:
            block2_in = block1_out

        block2_out = self._film(self.block2(block2_in), function_id, self.block2_film)
        out = self.block3(block2_out) + self.output_shift(function_id).view(x.shape[0], 1, 1)

        if self.tanh_act:
            out = nn.Tanh()(out)
        return out


class FunctionAwareSDFAutoEncoder(TransArticulatedBaseModule):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.n_validation = 0

        self.e_config = configs["evaluation"]
        self.e_config["eval_mesh_output_path"] = Path(self.e_config["eval_mesh_output_path"])
        self.e_config["eval_mesh_output_path"].mkdir(parents=True, exist_ok=True)

        sdf_configs = configs["SdfModelSpecs"]
        hidden_dim = sdf_configs["hidden_dim"]
        latent_dim = sdf_configs["latent_dim"]
        skip_connection = sdf_configs["skip_connection"]
        tanh_act = sdf_configs["tanh_act"]
        pn_hidden = sdf_configs["pn_hidden_dim"]

        self.encoder = Encoder(c_dim=latent_dim, hidden_dim=pn_hidden, plane_resolution=64)
        fa_config = configs.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.function_to_id = {name: idx for idx, name in enumerate(self.function_vocab)}
        self.num_functions = len(self.function_vocab)
        embed_dim = int(fa_config.get("embedding_dim", 64))
        self.use_latent_film = bool(fa_config.get("latent_film", True))
        self.use_latent_shift = bool(fa_config.get("latent_shift", self.use_latent_film))
        if fa_config.get("decoder_film", False):
            self.decoder = FunctionFiLMDecoder(
                latent_size=latent_dim,
                hidden_dim=hidden_dim,
                skip_connection=skip_connection,
                tanh_act=tanh_act,
                num_functions=self.num_functions,
                function_embedding_dim=embed_dim,
                film_scale=fa_config.get("decoder_film_scale", 0.1),
            )
        else:
            self.decoder = Decoder(latent_size=latent_dim, hidden_dim=hidden_dim,
                                   skip_connection=skip_connection, tanh_act=tanh_act)

        self.modulation_dim = latent_dim * 3
        latent_std = configs["latent_std"]
        hidden_dims = [self.modulation_dim] * 5
        self.vae_model = VAE(in_channels=self.modulation_dim, latent_dim=self.modulation_dim,
                             hidden_dims=hidden_dims, kl_std=latent_std)

        self.function_embedding = nn.Embedding(self.num_functions, embed_dim)
        self.input_film = nn.Linear(embed_dim, self.modulation_dim * 2)
        self.output_film = nn.Linear(embed_dim, self.modulation_dim * 2)
        self.latent_shift = nn.Embedding(self.num_functions, self.modulation_dim)
        self._init_function_modulators()

        weights = torch.ones(self.num_functions, dtype=torch.float32)
        for label, weight in fa_config.get("loss_weight_by_function", {}).items():
            if label in self.function_to_id:
                weights[self.function_to_id[label]] = float(weight)
        self.register_buffer("function_loss_weights", weights, persistent=False)

    def _init_function_modulators(self):
        nn.init.zeros_(self.input_film.weight)
        nn.init.zeros_(self.input_film.bias)
        nn.init.zeros_(self.output_film.weight)
        nn.init.zeros_(self.output_film.bias)
        nn.init.zeros_(self.latent_shift.weight)

    def initialize_from_sdf_checkpoint(self, ckpt_path: str | Path):
        base = SDFAutoEncoder.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        self.encoder.load_state_dict(base.encoder.state_dict())
        self.decoder.load_state_dict(base.decoder.state_dict(), strict=False)
        self.vae_model.load_state_dict(base.vae_model.state_dict())
        Log.info("Initialized function-aware SDF from %s", ckpt_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.configs["sdf_lr"])

    def _normalize_function_id(self, function_id, batch_size: int, device) -> torch.Tensor:
        if function_id is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(function_id):
            function_id = torch.tensor(function_id, dtype=torch.long, device=device)
        function_id = function_id.to(device=device, dtype=torch.long)
        if function_id.ndim == 0:
            function_id = function_id.repeat(batch_size)
        return function_id

    def _film(self, features: torch.Tensor, function_id: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        if not self.use_latent_film:
            return features
        emb = self.function_embedding(function_id)
        gamma_beta = layer(emb).view(features.shape[0], 2, features.shape[1], 1, 1)
        gamma = torch.tanh(gamma_beta[:, 0]) * 0.1
        beta = gamma_beta[:, 1] * 0.1
        return features * (1.0 + gamma) + beta

    def encode_plane_features(self, plane_features: torch.Tensor, function_id: torch.Tensor):
        conditioned_features = self._film(plane_features, function_id, self.input_film)
        out = self.vae_model(conditioned_features)
        reconstructed_plane_feature = self.decode_latent(out[-1], function_id)
        return [reconstructed_plane_feature, conditioned_features, out[2], out[3], out[-1]]

    def decode_latent(self, z: torch.Tensor, function_id=None) -> torch.Tensor:
        function_id = self._normalize_function_id(function_id, z.shape[0], z.device)
        if self.use_latent_shift:
            z = z + self.latent_shift(function_id)
        decoded = self.vae_model.decode(z)
        return self._film(decoded, function_id, self.output_film)

    def encode_latent_from_point_cloud(self, pc: torch.Tensor, function_id=None, deterministic: bool = True) -> torch.Tensor:
        function_id = self._normalize_function_id(function_id, pc.shape[0], pc.device)
        plane_features = self.encoder.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        conditioned_features = self._film(original_features, function_id, self.input_film)
        mu, log_var = self.vae_model.encode(conditioned_features)
        if deterministic:
            return mu
        return self.vae_model.reparameterize(mu, log_var)

    def decode_sdf_from_plane_features(self, plane_features: torch.Tensor, xyz: torch.Tensor, function_id=None):
        point_features = self.encoder.forward_with_plane_features(plane_features, xyz)
        decoder_input = torch.cat((xyz, point_features), dim=-1)
        if isinstance(self.decoder, FunctionFiLMDecoder):
            return self.decoder(decoder_input, function_id=function_id)
        return self.decoder(decoder_input)

    def _eikonal_loss(self, plane_features: torch.Tensor, function_id: torch.Tensor, xyz: torch.Tensor):
        weight = float(self.configs.get("function_aware", {}).get("eikonal_weight", 0.0))
        if weight <= 0 or not self.training:
            return xyz.new_tensor(0.0)

        count = int(self.configs.get("function_aware", {}).get("eikonal_sample_count", 1024))
        count = min(count, xyz.shape[1])
        idx = torch.randperm(xyz.shape[1], device=xyz.device)[:count]
        base_xyz = xyz[:, idx].detach()
        direction = torch.randn_like(base_xyz)
        direction = F.normalize(direction, dim=-1)
        eps = float(self.configs.get("function_aware", {}).get("finite_difference_eps", 0.01))

        pred_plus = self.decode_sdf_from_plane_features(
            plane_features, (base_xyz + eps * direction).clamp(-1.1, 1.1), function_id=function_id
        )
        pred_minus = self.decode_sdf_from_plane_features(
            plane_features, (base_xyz - eps * direction).clamp(-1.1, 1.1), function_id=function_id
        )
        directional_grad = (pred_plus - pred_minus).squeeze(-1) / (2.0 * eps)
        return (F.relu(directional_grad.abs() - 1.0) ** 2).mean() * weight

    def step(self, batch, batch_idx):
        xyz = batch["xyz"]
        gt = batch["gt_sdf"]
        pc = batch["point_cloud"]
        function_id = self._normalize_function_id(batch.get("function_id"), xyz.shape[0], xyz.device)

        plane_features = self.encoder.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.encode_plane_features(original_features, function_id)
        reconstructed_plane_feature, conditioned_features, mu, log_var, z = out

        point_features = self.encoder.forward_with_plane_features(reconstructed_plane_feature, xyz)
        decoder_input = torch.cat((xyz, point_features), dim=-1)
        if isinstance(self.decoder, FunctionFiLMDecoder):
            pred_sdf = self.decoder(decoder_input, function_id=function_id)
        else:
            pred_sdf = self.decoder(decoder_input)

        vae_loss = self.vae_model.loss_function(
            reconstructed_plane_feature, conditioned_features, mu, log_var, z,
            M_N=self.configs["kld_weight"],
        )

        sdf_loss_per = F.l1_loss(pred_sdf.squeeze(-1), gt.squeeze(-1), reduction="none")
        sdf_loss_per = sdf_loss_per.flatten(start_dim=1).mean(dim=1)
        weights = self.function_loss_weights[function_id]
        sdf_loss = (sdf_loss_per * weights).mean()

        plane_recon_weight = float(self.configs.get("function_aware", {}).get("plane_recon_weight", 0.0))
        plane_recon_loss = F.l1_loss(reconstructed_plane_feature, original_features) * plane_recon_weight
        eikonal_loss = self._eikonal_loss(reconstructed_plane_feature, function_id, xyz)

        loss = sdf_loss + vae_loss + plane_recon_loss + eikonal_loss
        return {
            "sdf_loss": sdf_loss,
            "vae_loss": vae_loss,
            "plane_recon_loss": plane_recon_loss,
            "eikonal_loss": eikonal_loss,
            "loss": loss,
            "reconstructed_plane_feature": reconstructed_plane_feature,
            "function_id": function_id,
            "z": z,
        }

    def training_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        self.log_dict({
            "sdf_loss": result["sdf_loss"],
            "vae_loss": result["vae_loss"],
            "plane_recon_loss": result["plane_recon_loss"],
            "eikonal_loss": result["eikonal_loss"],
            "loss": result["loss"],
        }, prog_bar=True, enable_graph=False)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        self.log_dict({
            "val_sdf_loss": result["sdf_loss"],
            "val_vae_loss": result["vae_loss"],
            "val_plane_recon_loss": result["plane_recon_loss"],
            "val_loss": result["loss"],
        }, prog_bar=False, enable_graph=False)

        if batch_idx == 0:
            self.n_validation += 1

        if self.global_rank != 0:
            return result["loss"]

        if batch_idx == 0 and self.n_validation % self.e_config["vis_epoch_freq"] == 0:
            batched_recon_latent = result["reconstructed_plane_feature"]
            function_ids = result["function_id"]
            evaluation_count = min(self.e_config["count"], batched_recon_latent.shape[0])
            screenshots = [np.random.randn(256, 256, 3) * 255 for _ in range(evaluation_count)]
            for batch_id in tqdm(range(evaluation_count), desc=f"Generating Mesh for Epoch = {batch_idx}"):
                output_mesh = self.e_config["eval_mesh_output_path"] / f"mesh_{self.trainer.current_epoch}_{batch_id}.ply"
                try:
                    MeshUtils.create_mesh(
                        self,
                        batched_recon_latent[[batch_id]],
                        output_mesh.as_posix(),
                        N=self.e_config["resolution"],
                        max_batch=self.e_config["max_batch"],
                        from_plane_features=True,
                        function_id=function_ids[[batch_id]],
                    )
                    mesh = trimesh.load(output_mesh)
                    screenshots[batch_id] = MeshUtils.generate_mesh_screenshot(mesh)
                except Exception as e:
                    Log.error(f"Error while generating mesh: {e}")
                    continue

            image = np.concatenate(screenshots, axis=1)
            if getattr(self, "logger", None) is not None and hasattr(self.logger, "log_image"):
                try:
                    self.logger.log_image(key="Image", images=[wandb.Image(image)])
                except Exception as e:
                    Log.error(f"Error while logging images: {e}")

        return result["loss"]
