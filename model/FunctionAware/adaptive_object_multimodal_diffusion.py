from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log

from .adaptive_multimodal_diffusion import cosine_beta_schedule, sinusoidal_embedding
from .functions import FUNCTION_VOCAB
from .sdf import FunctionAwareSDFAutoEncoder


class TokenConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.mean(dim=2)
        elif x.ndim == 2:
            x = x[:, None, :]
        return self.proj(x)


class AdaptiveObjectLatentDenoiser(nn.Module):
    def __init__(self, config, num_functions: int):
        super().__init__()
        m = config["model"]
        self.latent_dim = int(m.get("latent_dim", 768))
        self.hidden_dim = int(m.get("hidden_dim", 512))
        self.cond_dim = int(m.get("cond_dim", self.hidden_dim))
        self.time_dim = int(m.get("time_dim", 256))
        self.max_parts = int(m.get("max_parts", 16))
        dropout = float(m.get("dropout", 0.1))

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.function_embedding = nn.Embedding(num_functions, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_parts, self.hidden_dim)
        self.text_encoder = TokenConditionEncoder(
            int(m.get("text_dim", 1024)),
            self.hidden_dim,
            int(m.get("condition_hidden_dim", 512)),
        )
        self.image_encoder = TokenConditionEncoder(
            int(m.get("image_dim", 1408)),
            self.hidden_dim,
            int(m.get("condition_hidden_dim", 512)),
        )
        self.null_function = nn.Parameter(torch.zeros(self.hidden_dim))
        self.null_text = nn.Parameter(torch.zeros(self.hidden_dim))
        self.null_image = nn.Parameter(torch.zeros(self.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(m.get("n_heads", 8)),
            dim_feedforward=int(m.get("ffn_dim", self.hidden_dim * 4)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(m.get("depth", 6)))
        self.out = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.latent_dim))

    def _mask_or_null(self, value, keep, null_value):
        null = null_value.view(1, 1, -1).expand_as(value)
        return torch.where(keep[..., None].bool(), value, null)

    def _apply_drop(self, keep, drop_value):
        if torch.is_tensor(drop_value):
            while drop_value.ndim < keep.ndim:
                drop_value = drop_value[:, None]
            return keep & (~drop_value.to(device=keep.device, dtype=torch.bool))
        if bool(drop_value):
            return torch.zeros_like(keep)
        return keep

    def build_condition(self, batch, drop=None):
        device = batch["function_id"].device
        b, n = batch["function_id"].shape
        drop = drop or {}

        function_id = batch["function_id"].to(device=device, dtype=torch.long).clamp(0, self.function_embedding.num_embeddings - 1)
        function_keep = batch.get("mask", torch.ones(b, n, device=device)).to(device=device) > 0.5
        function_keep = self._apply_drop(function_keep, drop.get("function", False))
        function_cond = self.function_embedding(function_id)
        function_cond = self._mask_or_null(function_cond, function_keep, self.null_function)

        text_cond = self.text_encoder(batch["text"].to(device=device, dtype=torch.float32))
        text_keep = batch.get("has_text", torch.ones(b, n, device=device)).to(device=device) > 0.5
        text_keep = self._apply_drop(text_keep, drop.get("text", False))
        text_cond = self._mask_or_null(text_cond, text_keep, self.null_text)

        image_cond = self.image_encoder(batch["image"].to(device=device, dtype=torch.float32))
        image_keep = batch.get("has_image", torch.ones(b, n, device=device)).to(device=device) > 0.5
        image_keep = self._apply_drop(image_keep, drop.get("image", False))
        image_cond = self._mask_or_null(image_cond, image_keep, self.null_image)

        positions = torch.arange(n, device=device).clamp(max=self.max_parts - 1)
        position_cond = self.position_embedding(positions)[None].expand(b, n, -1)
        return function_cond + text_cond + image_cond + position_cond

    def forward(self, x_t, timesteps, batch, drop=None):
        b, n, _ = x_t.shape
        time = sinusoidal_embedding(timesteps, self.time_dim)
        time = self.time_mlp(time)[:, None, :].expand(b, n, -1)
        x = self.latent_proj(x_t) + time + self.build_condition(batch, drop=drop)
        key_padding = batch.get("mask", torch.ones(b, n, device=x.device)).to(device=x.device) < 0.5
        x = self.transformer(x, src_key_padding_mask=key_padding)
        return self.out(x)


class AdaptiveObjectMultimodalFunctionAwareDiffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        fa_config = config.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.num_functions = len(self.function_vocab)
        self.latent_dim = int(config["model"].get("latent_dim", 768))
        self.denoiser = AdaptiveObjectLatentDenoiser(config, self.num_functions)

        timesteps = int(config.get("diffusion", {}).get("timesteps", 1000))
        betas = cosine_beta_schedule(timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.num_timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_function_latent_stats(config.get("latent_normalization", {}))

        self.sdf = None
        sdf_path = config.get("evaluation", {}).get("sdf_model_path")
        if sdf_path:
            try:
                self.sdf = FunctionAwareSDFAutoEncoder.load_from_checkpoint(str(sdf_path), map_location="cpu")
                self.sdf.eval()
                self.sdf.requires_grad_(False)
            except Exception as exc:
                Log.warning("Could not load SDF checkpoint for evaluation: %s", exc)

    def register_function_latent_stats(self, stats_config):
        stats_path = stats_config.get("stats_path")
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path, allow_pickle=True)
            mean = torch.from_numpy(stats["mean"].astype(np.float32))
            std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-6))
        else:
            mean = torch.zeros(self.num_functions, self.latent_dim)
            std = torch.ones(self.num_functions, self.latent_dim)
        self.register_buffer("function_latent_mean", mean)
        self.register_buffer("function_latent_std", std)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=float(self.config.get("lr", 2.0e-4)),
            weight_decay=float(self.config.get("weight_decay", 1.0e-4)),
        )

    def q_sample(self, x_start, timesteps, noise):
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        return torch.sqrt(alpha) * x_start + torch.sqrt(1.0 - alpha) * noise

    def _training_drop(self, batch_size, n_parts, device):
        cfg = self.config.get("condition_dropout", {})
        uncond = torch.rand(batch_size, device=device) < float(cfg.get("unconditional_prob", 0.15))
        token_shape = (batch_size, n_parts)
        text = uncond[:, None] | (torch.rand(token_shape, device=device) < float(cfg.get("text_prob", 0.10)))
        image = uncond[:, None] | (torch.rand(token_shape, device=device) < float(cfg.get("image_prob", 0.10)))
        function = uncond[:, None] & bool(cfg.get("drop_function_for_unconditional", False))
        return {"text": text, "image": image, "function": function}

    def _masked_loss(self, pred, target, mask):
        loss = (pred - target).pow(2)
        loss = loss * mask[..., None].float()
        return loss.sum() / torch.clamp(mask.sum() * pred.shape[-1], min=1.0)

    def training_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        drop = self._training_drop(x0.shape[0], x0.shape[1], x0.device)
        pred_noise = self.denoiser(x_t, t, batch, drop=drop)
        loss = self._masked_loss(pred_noise, noise, mask)
        self.log_dict({"loss": loss}, prog_bar=True, enable_graph=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(x_t, t, batch, drop={"text": False, "image": False, "function": False})
        loss = self._masked_loss(pred_noise, noise, mask)
        self.log("val_loss", loss, prog_bar=False, enable_graph=False)
        return loss

    def unnormalize_latent(self, z, function_id):
        function_id = function_id.to(z.device, dtype=torch.long).clamp(0, self.num_functions - 1)
        mean = self.function_latent_mean.to(z.device)[function_id]
        std = self.function_latent_std.to(z.device)[function_id]
        return z * std + mean

    def _predict_noise_cfg(self, x_t, t, batch, guidance_scale: float, drop_text: bool, drop_image: bool):
        cond = self.denoiser(x_t, t, batch, drop={"text": drop_text, "image": drop_image, "function": False})
        if guidance_scale == 1.0:
            return cond
        uncond = self.denoiser(x_t, t, batch, drop={"text": True, "image": True, "function": False})
        return uncond + guidance_scale * (cond - uncond)

    @torch.no_grad()
    def sample_normalized(
        self,
        batch,
        guidance_scale: float = 1.5,
        use_text: bool = True,
        use_image: bool = True,
        clip_denoised: float | None = 2.5,
    ):
        device = next(self.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        b, n = batch["function_id"].shape
        mask = batch.get("mask", torch.ones(b, n, device=device)).to(device=device)
        x = torch.randn(b, n, self.latent_dim, device=device) * mask[..., None]

        for t_value in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((b,), t_value, device=device, dtype=torch.long)
            eps = self._predict_noise_cfg(
                x,
                t,
                batch,
                guidance_scale=guidance_scale,
                drop_text=not use_text,
                drop_image=not use_image,
            )
            alpha_bar = self.alphas_cumprod[t_value]
            alpha_bar_prev = self.alphas_cumprod[t_value - 1] if t_value > 0 else torch.ones_like(alpha_bar)
            beta = self.betas[t_value]
            alpha = 1.0 - beta
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            if clip_denoised is not None and clip_denoised > 0:
                x0 = x0.clamp(-float(clip_denoised), float(clip_denoised))
            denom = torch.clamp(1.0 - alpha_bar, min=1e-12)
            mean = beta * torch.sqrt(alpha_bar_prev) / denom * x0 + (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / denom * x
            if t_value > 0:
                var = beta * (1.0 - alpha_bar_prev) / denom
                x = mean + torch.sqrt(torch.clamp(var, min=1e-20)) * torch.randn_like(x)
            else:
                x = mean
            x = x * mask[..., None]
        return x

    @torch.no_grad()
    def sample_latents(self, batch, **kwargs):
        normalized = self.sample_normalized(batch, **kwargs)
        return self.unnormalize_latent(normalized, batch["function_id"].to(normalized.device))
