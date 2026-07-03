from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log

from .functions import FUNCTION_VOCAB
from .sdf import FunctionAwareSDFAutoEncoder


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class PooledConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x):
        if x.ndim == 2:
            pooled = x
        elif x.ndim == 3:
            pooled = x.mean(dim=1)
        else:
            pooled = x.reshape(x.shape[0], -1)
        return self.proj(pooled)


class AdaptiveLatentDenoiser(nn.Module):
    def __init__(self, config, num_functions: int):
        super().__init__()
        m = config["model"]
        self.latent_dim = int(m.get("latent_dim", 768))
        self.hidden_dim = int(m.get("hidden_dim", 1024))
        self.cond_dim = int(m.get("cond_dim", 512))
        self.time_dim = int(m.get("time_dim", 256))
        self.num_functions = num_functions
        dropout = float(m.get("dropout", 0.1))

        self.function_embedding = nn.Embedding(num_functions, self.cond_dim)
        self.null_function = nn.Parameter(torch.zeros(self.cond_dim))

        self.text_encoder = PooledConditionEncoder(
            int(m.get("text_dim", 1024)),
            self.cond_dim,
            int(m.get("condition_hidden_dim", 512)),
        )
        self.image_encoder = PooledConditionEncoder(
            int(m.get("image_dim", 1408)),
            self.cond_dim,
            int(m.get("condition_hidden_dim", 512)),
        )
        self.null_text = nn.Parameter(torch.zeros(self.cond_dim))
        self.null_image = nn.Parameter(torch.zeros(self.cond_dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.in_proj = nn.Linear(self.latent_dim + self.hidden_dim + self.cond_dim, self.hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(self.hidden_dim, dropout) for _ in range(int(m.get("depth", 8)))])
        self.out = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.latent_dim))

    def _mask_or_null(self, value, keep, null_value):
        null = null_value.view(1, -1).expand_as(value)
        return torch.where(keep[:, None].bool(), value, null)

    def _apply_drop(self, keep, drop_value):
        if torch.is_tensor(drop_value):
            return keep & (~drop_value.to(device=keep.device, dtype=torch.bool))
        if bool(drop_value):
            return torch.zeros_like(keep)
        return keep

    def build_condition(self, batch, drop=None):
        anchor = batch.get("latent_code", batch.get("function_id", batch.get("text")))
        if anchor is None:
            raise KeyError("Condition batch must contain at least one of latent_code/function_id/text.")
        device = anchor.device
        batch_size = anchor.shape[0]
        drop = drop or {}

        function_id = batch.get("function_id")
        if function_id is None:
            function_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        function_id = function_id.to(device=device, dtype=torch.long)
        function_keep = torch.ones(batch_size, device=device, dtype=torch.bool)
        function_keep = self._apply_drop(function_keep, drop.get("function", False))
        function_cond = self.function_embedding(function_id.clamp(0, self.num_functions - 1))
        function_cond = self._mask_or_null(function_cond, function_keep, self.null_function)

        text_cond = self.text_encoder(batch["text"].to(device=device, dtype=torch.float32))
        text_keep = batch.get("has_text", torch.ones(batch_size, device=device)).to(device=device) > 0.5
        text_keep = self._apply_drop(text_keep, drop.get("text", False))
        text_cond = self._mask_or_null(text_cond, text_keep, self.null_text)

        image_cond = self.image_encoder(batch["image"].to(device=device, dtype=torch.float32))
        image_keep = batch.get("has_image", torch.ones(batch_size, device=device)).to(device=device) > 0.5
        image_keep = self._apply_drop(image_keep, drop.get("image", False))
        image_cond = self._mask_or_null(image_cond, image_keep, self.null_image)

        return function_cond + text_cond + image_cond

    def forward(self, x_t, timesteps, batch, drop=None):
        t = sinusoidal_embedding(timesteps, self.time_dim)
        t = self.time_mlp(t)
        cond = self.build_condition(batch, drop=drop)
        x = torch.cat([x_t, t, cond], dim=-1)
        x = self.in_proj(x)
        return self.out(self.blocks(x))


class AdaptiveMultimodalFunctionAwareDiffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        fa_config = config.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.num_functions = len(self.function_vocab)

        self.model_config = config["model"]
        self.latent_dim = int(self.model_config.get("latent_dim", 768))
        self.objective = config.get("diffusion", {}).get("objective", "pred_noise")
        if self.objective != "pred_noise":
            raise ValueError("Adaptive diffusion currently supports objective=pred_noise only.")

        self.denoiser = AdaptiveLatentDenoiser(config, self.num_functions)

        timesteps = int(config.get("diffusion", {}).get("timesteps", 1000))
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.register_latent_stats(config.get("latent_normalization", {}))

        self.sdf = None
        sdf_path = config.get("evaluation", {}).get("sdf_model_path")
        if sdf_path:
            try:
                self.sdf = FunctionAwareSDFAutoEncoder.load_from_checkpoint(str(sdf_path), map_location="cpu")
                self.sdf.eval()
                self.sdf.requires_grad_(False)
            except Exception as exc:
                Log.warning("Could not load SDF checkpoint for evaluation: %s", exc)

    def register_latent_stats(self, stats_config):
        stats_path = stats_config.get("stats_path")
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path, allow_pickle=True)
            mean = torch.from_numpy(stats["mean"].astype(np.float32))
            std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-6))
        else:
            mean = torch.zeros(self.latent_dim)
            std = torch.ones(self.latent_dim)
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_std", std)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=float(self.config.get("lr", 2.0e-4)),
            weight_decay=float(self.config.get("weight_decay", 1.0e-4)),
        )

    def q_sample(self, x_start, timesteps, noise):
        return (
            self.sqrt_alphas_cumprod[timesteps, None] * x_start
            + self.sqrt_one_minus_alphas_cumprod[timesteps, None] * noise
        )

    def _training_drop(self, batch_size, device):
        cfg = self.config.get("condition_dropout", {})
        uncond = torch.rand(batch_size, device=device) < float(cfg.get("unconditional_prob", 0.15))
        text = uncond | (torch.rand(batch_size, device=device) < float(cfg.get("text_prob", 0.10)))
        image = uncond | (torch.rand(batch_size, device=device) < float(cfg.get("image_prob", 0.10)))
        function = uncond & bool(cfg.get("drop_function_for_unconditional", False))
        return {"text": text, "image": image, "function": function}

    def training_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        drop = self._training_drop(x0.shape[0], x0.device)
        pred_noise = self.denoiser(x_t, t, batch, drop=drop)
        loss = F.mse_loss(pred_noise, noise)
        self.log_dict({"loss": loss}, prog_bar=True, enable_graph=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(x_t, t, batch, drop={"text": False, "image": False, "function": False})
        loss = F.mse_loss(pred_noise, noise)
        self.log("val_loss", loss, prog_bar=False, enable_graph=False)
        return loss

    def unnormalize_latent(self, z):
        return z * self.latent_std.to(z.device) + self.latent_mean.to(z.device)

    def normalize_latent(self, z):
        return (z - self.latent_mean.to(z.device)) / self.latent_std.to(z.device)

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
        steps: int = 80,
        guidance_scale: float = 1.5,
        use_text: bool = True,
        use_image: bool = True,
        sampler: str = "ddim",
        clip_denoised: float | None = 5.0,
    ):
        device = next(self.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        b = batch["function_id"].shape[0]
        x = torch.randn(b, self.latent_dim, device=device)

        if sampler == "ddpm":
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
                posterior_mean = (
                    beta * torch.sqrt(alpha_bar_prev) / denom * x0
                    + (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / denom * x
                )
                if t_value > 0:
                    posterior_var = beta * (1.0 - alpha_bar_prev) / denom
                    x = posterior_mean + torch.sqrt(torch.clamp(posterior_var, min=1e-20)) * torch.randn_like(x)
                else:
                    x = posterior_mean
            return x

        if sampler != "ddim":
            raise ValueError(f"Unsupported sampler: {sampler}")

        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()
        for idx, t_value in enumerate(times):
            t = torch.full((b,), int(t_value.item()), device=device, dtype=torch.long)
            eps = self._predict_noise_cfg(
                x,
                t,
                batch,
                guidance_scale=guidance_scale,
                drop_text=not use_text,
                drop_image=not use_image,
            )
            alpha = self.alphas_cumprod[t_value]
            alpha_prev = self.alphas_cumprod[times[idx + 1]] if idx + 1 < len(times) else torch.ones_like(alpha)
            x0 = (x - torch.sqrt(1.0 - alpha) * eps) / torch.sqrt(alpha)
            if clip_denoised is not None and clip_denoised > 0:
                x0 = x0.clamp(-float(clip_denoised), float(clip_denoised))
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps
        return x

    @torch.no_grad()
    def sample_latents(self, batch, **kwargs):
        return self.unnormalize_latent(self.sample_normalized(batch, **kwargs))
