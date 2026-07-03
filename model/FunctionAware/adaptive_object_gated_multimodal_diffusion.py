from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log

from .adaptive_multimodal_diffusion import cosine_beta_schedule, sinusoidal_embedding
from .functions import FUNCTION_TO_ID, FUNCTION_VOCAB
from .sdf import FunctionAwareSDFAutoEncoder


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (values * weights[..., None]).sum(dim=1) / denom


def gather_object_sequence(values: torch.Tensor, availability: torch.Tensor, mask: torch.Tensor | None):
    if values.ndim == 3:
        has_any = availability.to(device=values.device) > 0.5
        return values, has_any
    if values.ndim != 4:
        raise ValueError(f"Expected condition tensor with ndim 3 or 4, got {values.ndim}")

    availability = availability.to(device=values.device) > 0.5
    if mask is None:
        valid = availability
    else:
        valid = availability & (mask.to(device=values.device) > 0.5)
    weights = valid.float()
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled = (values * weights[:, :, None, None]).sum(dim=1) / denom[:, :, None]
    has_any = valid.any(dim=1)
    return pooled, has_any


class SequenceConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pooled_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x[:, None, :]
        tokens = self.token_proj(x)
        pooled = self.pooled_proj(tokens.mean(dim=1))
        return tokens, pooled


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.text_norm = nn.LayerNorm(hidden_dim)
        self.text_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.image_norm = nn.LayerNorm(hidden_dim)
        self.image_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.text_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.image_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(hidden_dim, ffn_dim, dropout)

    def _cross_with_gate(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pooled_context: torch.Tensor,
        available: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
        gate_net: nn.Sequential,
    ) -> torch.Tensor:
        query = norm(x)
        attended, _ = attn(query, context, context, need_weights=False)
        pooled = pooled_context[:, None, :].expand(-1, x.shape[1], -1)
        gate_input = torch.cat([query, pooled, query * pooled], dim=-1)
        gate = torch.sigmoid(gate_net(gate_input))
        gate = gate * available[:, None, None].float()
        return x + self.dropout(attended) * gate

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        image_pooled: torch.Tensor,
        has_text: torch.Tensor,
        has_image: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = self.self_norm(x)
        attended, _ = self.self_attn(
            residual,
            residual,
            residual,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        x = self._cross_with_gate(x, text_tokens, text_pooled, has_text, self.text_attn, self.text_norm, self.text_gate)
        x = self._cross_with_gate(x, image_tokens, image_pooled, has_image, self.image_attn, self.image_norm, self.image_gate)
        return self.ff(x)


class AdaptiveObjectGatedLatentDenoiser(nn.Module):
    def __init__(self, config, num_functions: int):
        super().__init__()
        m = config["model"]
        self.latent_dim = int(m.get("latent_dim", 768))
        self.hidden_dim = int(m.get("hidden_dim", 512))
        self.time_dim = int(m.get("time_dim", 256))
        self.max_parts = int(m.get("max_parts", 16))
        self.default_function_id = int(FUNCTION_TO_ID.get("static_part", 0))

        dropout = float(m.get("dropout", 0.1))
        n_heads = int(m.get("n_heads", 8))
        ffn_dim = int(m.get("ffn_dim", self.hidden_dim * 4))
        condition_hidden = int(m.get("condition_hidden_dim", self.hidden_dim))

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.function_embedding = nn.Embedding(num_functions, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_parts, self.hidden_dim)

        self.text_encoder = SequenceConditionEncoder(int(m.get("text_dim", 1024)), self.hidden_dim)
        self.image_encoder = SequenceConditionEncoder(int(m.get("image_dim", 768)), self.hidden_dim)
        self.object_context = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, condition_hidden),
            nn.SiLU(),
            nn.Linear(condition_hidden, self.hidden_dim),
        )

        self.object_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_text_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_image_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_text_pooled = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.null_image_pooled = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.null_function = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.blocks = nn.ModuleList(
            [GatedCrossAttentionBlock(self.hidden_dim, n_heads, ffn_dim, dropout) for _ in range(int(m.get("depth", 6)))]
        )
        self.out = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.latent_dim))

    def _drop_mask(self, available: torch.Tensor, drop_value):
        keep = available > 0.5
        if torch.is_tensor(drop_value):
            keep = keep & (~drop_value.to(device=keep.device, dtype=torch.bool))
        elif bool(drop_value):
            keep = torch.zeros_like(keep)
        return keep

    def _apply_null_part_tokens(self, values: torch.Tensor, keep: torch.Tensor):
        null = self.null_function.expand(values.shape[0], values.shape[1], -1)
        return torch.where(keep[..., None], values, null)

    def _encode_modality(self, encoder, values, availability, mask, drop_value, null_token, null_pooled):
        sequence, has_any = gather_object_sequence(values, availability, mask)
        encoded_tokens, encoded_pooled = encoder(sequence.float())
        keep = self._drop_mask(has_any, drop_value)
        tokens = torch.where(
            keep[:, None, None],
            encoded_tokens,
            null_token.expand(encoded_tokens.shape[0], encoded_tokens.shape[1], -1),
        )
        pooled = torch.where(keep[:, None], encoded_pooled, null_pooled.expand(encoded_pooled.shape[0], -1))
        return tokens, pooled, keep

    def forward(self, x_t, timesteps, batch, drop=None):
        drop = drop or {}
        device = x_t.device
        batch_size, n_parts, _ = x_t.shape

        mask = batch.get("mask", torch.ones(batch_size, n_parts, device=device)).to(device=device) > 0.5
        function_id = batch["function_id"].to(device=device, dtype=torch.long).clamp(min=0, max=self.function_embedding.num_embeddings - 1)
        function_tokens = self.function_embedding(function_id)
        if "function" in drop:
            function_keep = ~(drop["function"].to(device=device, dtype=torch.bool)) if torch.is_tensor(drop["function"]) else mask
            function_tokens = self._apply_null_part_tokens(function_tokens, function_keep & mask)

        time = sinusoidal_embedding(timesteps, self.time_dim)
        time = self.time_mlp(time)[:, None, :].expand(batch_size, n_parts, -1)
        positions = torch.arange(n_parts, device=device).clamp(max=self.max_parts - 1)
        position_tokens = self.position_embedding(positions)[None].expand(batch_size, n_parts, -1)

        text_tokens, text_pooled, has_text = self._encode_modality(
            self.text_encoder,
            batch["text"].to(device=device),
            batch.get("has_text", torch.ones(batch_size, n_parts, device=device)).to(device=device),
            mask,
            drop.get("text", False),
            self.null_text_token,
            self.null_text_pooled,
        )
        image_tokens, image_pooled, has_image = self._encode_modality(
            self.image_encoder,
            batch["image"].to(device=device),
            batch.get("has_image", torch.ones(batch_size, n_parts, device=device)).to(device=device),
            mask,
            drop.get("image", False),
            self.null_image_token,
            self.null_image_pooled,
        )

        part_tokens = self.latent_proj(x_t) + time + function_tokens + position_tokens
        pooled_function = _masked_mean(function_tokens, mask.float())
        object_context = self.object_context(torch.cat([pooled_function, text_pooled, image_pooled], dim=-1))
        object_token = self.object_token.expand(batch_size, 1, -1) + object_context[:, None, :]

        tokens = torch.cat([object_token, part_tokens], dim=1)
        key_padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, device=device, dtype=torch.bool), ~mask],
            dim=1,
        )

        for block in self.blocks:
            tokens = block(tokens, text_tokens, image_tokens, text_pooled, image_pooled, has_text, has_image, key_padding_mask)

        return self.out(tokens[:, 1:, :])


class AdaptiveObjectGatedMultimodalFunctionAwareDiffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        fa_config = config.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.num_functions = len(self.function_vocab)
        self.latent_dim = int(config["model"].get("latent_dim", 768))
        self.denoiser = AdaptiveObjectGatedLatentDenoiser(config, self.num_functions)

        timesteps = int(config.get("diffusion", {}).get("timesteps", 1000))
        betas = cosine_beta_schedule(timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.num_timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_function_latent_stats(config.get("latent_normalization", {}))
        self.register_function_loss_weights(fa_config.get("diffusion_loss_weight_by_function", {}))

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

    def register_function_loss_weights(self, overrides):
        weights = torch.ones(self.num_functions, dtype=torch.float32)
        for label, weight in overrides.items():
            index = FUNCTION_TO_ID.get(label)
            if index is not None and index < self.num_functions:
                weights[index] = float(weight)
        self.register_buffer("function_loss_weights", weights, persistent=False)

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
        text = uncond | (torch.rand(batch_size, device=device) < float(cfg.get("text_prob", 0.10)))
        image = uncond | (torch.rand(batch_size, device=device) < float(cfg.get("image_prob", 0.10)))
        function = torch.zeros((batch_size, n_parts), device=device, dtype=torch.bool)
        if bool(cfg.get("drop_function_for_unconditional", False)):
            function = uncond[:, None].expand(-1, n_parts).clone()
        return {"text": text, "image": image, "function": function}

    def _masked_loss(self, pred, target, mask, function_id):
        weights = self.function_loss_weights.to(mask.device)[function_id.clamp(min=0, max=self.num_functions - 1)]
        weighted_mask = mask.float() * weights.float()
        loss = (pred - target).pow(2) * weighted_mask[..., None]
        denom = torch.clamp(weighted_mask.sum() * pred.shape[-1], min=1.0)
        return loss.sum() / denom

    def training_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        drop = self._training_drop(x0.shape[0], x0.shape[1], x0.device)
        pred_noise = self.denoiser(x_t, t, batch, drop=drop)
        loss = self._masked_loss(pred_noise, noise, mask, batch["function_id"].long())
        self.log_dict({"loss": loss}, prog_bar=True, enable_graph=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(x_t, t, batch, drop={"text": False, "image": False, "function": False})
        loss = self._masked_loss(pred_noise, noise, mask, batch["function_id"].long())
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
