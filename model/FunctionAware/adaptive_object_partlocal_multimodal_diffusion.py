from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.base import TransArticulatedBaseModule
from utils.mylogging import Log

from .adaptive_multimodal_diffusion import cosine_beta_schedule, sinusoidal_embedding
from .adaptive_object_gated_multimodal_diffusion import _masked_mean, gather_object_sequence
from .adaptive_object_routed_multimodal_diffusion import FunctionAwareOutputHead
from .functions import FUNCTION_TO_ID, FUNCTION_VOCAB
from .sdf import FunctionAwareSDFAutoEncoder


class PartTextConditionEncoder(nn.Module):
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
        if x.ndim == 3:
            x = x[:, :, None, :]
        if x.ndim != 4:
            raise ValueError(f"Expected text tensor with ndim 3 or 4, got {x.ndim}")
        b, n, t, d = x.shape
        tokens = self.token_proj(x.reshape(b * n, t, d)).reshape(b, n, t, -1)
        pooled = self.pooled_proj(tokens.mean(dim=2))
        return tokens, pooled


class ObjectImageConditionEncoder(nn.Module):
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
        if x.ndim != 3:
            raise ValueError(f"Expected image tensor with ndim 2 or 3 after object gather, got {x.ndim}")
        tokens = self.token_proj(x)
        pooled = self.pooled_proj(tokens.mean(dim=1))
        return tokens, pooled


class PartLocalConditionBlock(nn.Module):
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
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _apply_part_text(
        self,
        state: torch.Tensor,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        has_text: torch.Tensor,
    ) -> torch.Tensor:
        part_state = state[:, 1:, :]
        b, n, h = part_state.shape
        t = text_tokens.shape[2]

        query = self.text_norm(part_state).reshape(b * n, 1, h)
        context = text_tokens.reshape(b * n, t, h)
        attended, _ = self.text_attn(query, context, context, need_weights=False)
        attended = attended.reshape(b, n, h)

        pooled = text_pooled
        gate_input = torch.cat([part_state, pooled, part_state * pooled], dim=-1)
        gate = torch.sigmoid(self.text_gate(gate_input)) * has_text[:, :, None].float()

        updated = part_state + self.dropout(attended) * gate
        return torch.cat([state[:, :1, :], updated], dim=1)

    def _apply_object_image(
        self,
        state: torch.Tensor,
        image_tokens: torch.Tensor,
        image_pooled: torch.Tensor,
        has_image: torch.Tensor,
    ) -> torch.Tensor:
        query = self.image_norm(state)
        attended, _ = self.image_attn(query, image_tokens, image_tokens, need_weights=False)
        pooled = image_pooled[:, None, :].expand(-1, state.shape[1], -1)
        gate_input = torch.cat([state, pooled, state * pooled], dim=-1)
        gate = torch.sigmoid(self.image_gate(gate_input)) * has_image[:, None, None].float()
        return state + self.dropout(attended) * gate

    def forward(
        self,
        state: torch.Tensor,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        has_text: torch.Tensor,
        image_tokens: torch.Tensor,
        image_pooled: torch.Tensor,
        has_image: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        use_text: bool,
        use_image: bool,
    ) -> torch.Tensor:
        residual = self.self_norm(state)
        attended, _ = self.self_attn(
            residual,
            residual,
            residual,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        state = state + self.dropout(attended)
        if use_text:
            state = self._apply_part_text(state, text_tokens, text_pooled, has_text)
        if use_image:
            state = self._apply_object_image(state, image_tokens, image_pooled, has_image)
        return state + self.ff(state)


class PartLocalObjectLatentDenoiser(nn.Module):
    def __init__(self, config, num_functions: int):
        super().__init__()
        m = config["model"]
        self.latent_dim = int(m.get("latent_dim", 768))
        self.hidden_dim = int(m.get("hidden_dim", 512))
        self.time_dim = int(m.get("time_dim", 256))
        self.max_parts = int(m.get("max_parts", 16))

        dropout = float(m.get("dropout", 0.1))
        n_heads = int(m.get("n_heads", 8))
        ffn_dim = int(m.get("ffn_dim", self.hidden_dim * 4))
        condition_hidden = int(m.get("condition_hidden_dim", self.hidden_dim))
        base_depth = int(m.get("depth", 5))
        expert_depth = int(m.get("expert_depth", 2))
        fusion_depth = int(m.get("fusion_depth", expert_depth))

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.function_embedding = nn.Embedding(num_functions, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_parts, self.hidden_dim)

        self.text_encoder = PartTextConditionEncoder(int(m.get("text_dim", 1024)), self.hidden_dim)
        self.image_encoder = ObjectImageConditionEncoder(int(m.get("image_dim", 768)), self.hidden_dim)
        self.function_context = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, condition_hidden),
            nn.SiLU(),
            nn.Linear(condition_hidden, self.hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=base_depth)

        self.text_blocks = nn.ModuleList(
            [PartLocalConditionBlock(self.hidden_dim, n_heads, ffn_dim, dropout) for _ in range(expert_depth)]
        )
        self.image_blocks = nn.ModuleList(
            [PartLocalConditionBlock(self.hidden_dim, n_heads, ffn_dim, dropout) for _ in range(expert_depth)]
        )
        self.fusion_blocks = nn.ModuleList(
            [PartLocalConditionBlock(self.hidden_dim, n_heads, ffn_dim, dropout) for _ in range(fusion_depth)]
        )

        self.object_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_text_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))
        self.null_text_pooled = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_image_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.null_image_pooled = nn.Parameter(torch.zeros(1, self.hidden_dim))

        film_scale = float(m.get("output_film_scale", 0.08))
        self.uncond_head = FunctionAwareOutputHead(self.hidden_dim, self.latent_dim, num_functions, film_scale)
        self.text_head = FunctionAwareOutputHead(self.hidden_dim, self.latent_dim, num_functions, film_scale)
        self.image_head = FunctionAwareOutputHead(self.hidden_dim, self.latent_dim, num_functions, film_scale)
        self.fused_head = FunctionAwareOutputHead(self.hidden_dim, self.latent_dim, num_functions, film_scale)

    def _encode_text(self, batch, mask, device):
        values = batch["text"].to(device=device, dtype=torch.float32)
        tokens, pooled = self.text_encoder(values)
        b, n = mask.shape
        available = batch.get("has_text", torch.ones(b, n, device=device)).to(device=device) > 0.5
        keep = available & mask
        tokens = torch.where(keep[:, :, None, None], tokens, self.null_text_token.expand_as(tokens))
        pooled = torch.where(keep[:, :, None], pooled, self.null_text_pooled.expand_as(pooled))
        return tokens, pooled, keep

    def _encode_image(self, batch, mask, device):
        values = batch["image"].to(device=device, dtype=torch.float32)
        availability = batch.get("has_image", torch.ones(mask.shape, device=device)).to(device=device)
        sequence, has_any = gather_object_sequence(values, availability, mask)
        tokens, pooled = self.image_encoder(sequence)
        tokens = torch.where(has_any[:, None, None], tokens, self.null_image_token.expand_as(tokens))
        pooled = torch.where(has_any[:, None], pooled, self.null_image_pooled.expand_as(pooled))
        return tokens, pooled, has_any

    def _run_blocks(
        self,
        state: torch.Tensor,
        blocks: nn.ModuleList,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        has_text: torch.Tensor,
        image_tokens: torch.Tensor,
        image_pooled: torch.Tensor,
        has_image: torch.Tensor,
        key_padding_mask: torch.Tensor,
        use_text: bool,
        use_image: bool,
    ):
        for block in blocks:
            state = block(
                state,
                text_tokens,
                text_pooled,
                has_text,
                image_tokens,
                image_pooled,
                has_image,
                key_padding_mask,
                use_text=use_text,
                use_image=use_image,
            )
        return state

    def forward(self, x_t, timesteps, batch):
        device = x_t.device
        batch_size, n_parts, _ = x_t.shape
        mask = batch.get("mask", torch.ones(batch_size, n_parts, device=device)).to(device=device) > 0.5
        function_id = batch["function_id"].to(device=device, dtype=torch.long).clamp(
            min=0,
            max=self.function_embedding.num_embeddings - 1,
        )

        function_tokens = self.function_embedding(function_id)
        time = sinusoidal_embedding(timesteps, self.time_dim)
        time = self.time_mlp(time)[:, None, :].expand(batch_size, n_parts, -1)
        positions = torch.arange(n_parts, device=device).clamp(max=self.max_parts - 1)
        position_tokens = self.position_embedding(positions)[None].expand(batch_size, n_parts, -1)

        text_tokens, text_pooled, has_text = self._encode_text(batch, mask, device)
        image_tokens, image_pooled, has_image = self._encode_image(batch, mask, device)

        part_tokens = self.latent_proj(x_t) + time + function_tokens + position_tokens
        pooled_function = _masked_mean(function_tokens, mask.float())
        object_token = self.object_token.expand(batch_size, 1, -1) + self.function_context(pooled_function)[:, None, :]

        state = torch.cat([object_token, part_tokens], dim=1)
        key_padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, device=device, dtype=torch.bool), ~mask],
            dim=1,
        )
        base_state = self.backbone(state, src_key_padding_mask=key_padding_mask)

        text_state = self._run_blocks(
            base_state.clone(),
            self.text_blocks,
            text_tokens,
            text_pooled,
            has_text,
            image_tokens,
            image_pooled,
            has_image,
            key_padding_mask,
            use_text=True,
            use_image=False,
        )
        image_state = self._run_blocks(
            base_state.clone(),
            self.image_blocks,
            text_tokens,
            text_pooled,
            has_text,
            image_tokens,
            image_pooled,
            has_image,
            key_padding_mask,
            use_text=False,
            use_image=True,
        )
        fused_state = self._run_blocks(
            base_state.clone(),
            self.fusion_blocks,
            text_tokens,
            text_pooled,
            has_text,
            image_tokens,
            image_pooled,
            has_image,
            key_padding_mask,
            use_text=True,
            use_image=True,
        )

        return {
            "unconditional": self.uncond_head(base_state[:, 1:, :], function_id),
            "text": self.text_head(text_state[:, 1:, :], function_id),
            "image": self.image_head(image_state[:, 1:, :], function_id),
            "text_image": self.fused_head(fused_state[:, 1:, :], function_id),
        }


class AdaptiveObjectPartLocalMultimodalFunctionAwareDiffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        fa_config = config.get("function_aware", {})
        self.function_vocab = fa_config.get("vocab", FUNCTION_VOCAB)
        self.num_functions = len(self.function_vocab)
        self.latent_dim = int(config["model"].get("latent_dim", 768))
        self.objective = config.get("diffusion", {}).get("objective", "pred_v")
        if self.objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"Unsupported diffusion objective: {self.objective}")

        self.denoiser = PartLocalObjectLatentDenoiser(config, self.num_functions)

        timesteps = int(config.get("diffusion", {}).get("timesteps", 1000))
        betas = cosine_beta_schedule(timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.num_timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_function_latent_stats(config.get("latent_normalization", {}))
        self.register_branch_function_loss_weights(fa_config.get("diffusion_loss_weight_by_branch", {}))

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

    def register_branch_function_loss_weights(self, overrides):
        for branch_name in ["unconditional", "text", "image", "text_image"]:
            weights = torch.ones(self.num_functions, dtype=torch.float32)
            for label, weight in overrides.get(branch_name, {}).items():
                index = FUNCTION_TO_ID.get(label)
                if index is not None and index < self.num_functions:
                    weights[index] = float(weight)
            self.register_buffer(f"function_loss_weights_{branch_name}", weights, persistent=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=float(self.config.get("lr", 2.0e-4)),
            weight_decay=float(self.config.get("weight_decay", 1.0e-4)),
        )

    def q_sample(self, x_start, timesteps, noise):
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        return torch.sqrt(alpha) * x_start + torch.sqrt(1.0 - alpha) * noise

    def _target(self, x0, noise, timesteps):
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        if self.objective == "pred_noise":
            return noise
        if self.objective == "pred_x0":
            return x0
        return torch.sqrt(alpha) * noise - torch.sqrt(1.0 - alpha) * x0

    def _x0_and_eps_from_model_output(self, x_t, timesteps, model_output):
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)
        if self.objective == "pred_noise":
            eps = model_output
            x0 = (x_t - sqrt_one_minus_alpha * eps) / torch.clamp(sqrt_alpha, min=1e-12)
        elif self.objective == "pred_x0":
            x0 = model_output
            eps = (x_t - sqrt_alpha * x0) / torch.clamp(sqrt_one_minus_alpha, min=1e-12)
        else:
            v = model_output
            x0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v
            eps = sqrt_one_minus_alpha * x_t + sqrt_alpha * v
        return x0, eps

    def _masked_loss(self, pred, target, mask, function_id, branch_name: str):
        branch_weights = getattr(self, f"function_loss_weights_{branch_name}").to(mask.device)
        weights = branch_weights[function_id.clamp(min=0, max=self.num_functions - 1)]
        weighted_mask = mask.float() * weights.float()
        loss = (pred - target).pow(2) * weighted_mask[..., None]
        denom = torch.clamp(weighted_mask.sum() * pred.shape[-1], min=1.0)
        return loss.sum() / denom

    def _agreement_loss(self, pred_a, pred_b, mask):
        diff = (pred_a - pred_b).pow(2) * mask[..., None].float()
        denom = torch.clamp(mask.sum() * pred_a.shape[-1], min=1.0)
        return diff.sum() / denom

    def _branch_weights(self):
        loss_config = self.config.get("branch_loss", {})
        return {
            "unconditional": float(loss_config.get("unconditional_weight", 0.10)),
            "text": float(loss_config.get("text_weight", 0.90)),
            "image": float(loss_config.get("image_weight", 0.70)),
            "text_image": float(loss_config.get("text_image_weight", 0.45)),
            "agreement": float(loss_config.get("agreement_weight", 0.03)),
        }

    def training_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        target = self._target(x0, noise, t)
        preds = self.denoiser(x_t, t, batch)

        losses = {
            branch_name: self._masked_loss(preds[branch_name], target, mask, batch["function_id"].long(), branch_name)
            for branch_name in ["unconditional", "text", "image", "text_image"]
        }
        agreement_target = 0.5 * (preds["text"] + preds["image"])
        agreement_loss = self._agreement_loss(preds["text_image"], agreement_target, mask)

        weights = self._branch_weights()
        total_loss = (
            weights["unconditional"] * losses["unconditional"]
            + weights["text"] * losses["text"]
            + weights["image"] * losses["image"]
            + weights["text_image"] * losses["text_image"]
            + weights["agreement"] * agreement_loss
        )

        self.log_dict(
            {
                "loss": total_loss,
                "loss_uncond": losses["unconditional"],
                "loss_text": losses["text"],
                "loss_image": losses["image"],
                "loss_text_image": losses["text_image"],
                "loss_agreement": agreement_loss,
            },
            prog_bar=True,
            enable_graph=False,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        x0 = batch["latent_code"].float()
        mask = batch["mask"].float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        x_t = self.q_sample(x0, t, noise)
        target = self._target(x0, noise, t)
        preds = self.denoiser(x_t, t, batch)
        val_loss = self._masked_loss(preds["text_image"], target, mask, batch["function_id"].long(), "text_image")
        self.log("val_loss", val_loss, prog_bar=False, enable_graph=False)
        return val_loss

    def unnormalize_latent(self, z, function_id):
        function_id = function_id.to(z.device, dtype=torch.long).clamp(0, self.num_functions - 1)
        mean = self.function_latent_mean.to(z.device)[function_id]
        std = self.function_latent_std.to(z.device)[function_id]
        return z * std + mean

    def _choose_branch(self, use_text: bool, use_image: bool) -> str:
        if use_text and use_image:
            return "text_image"
        if use_text:
            return "text"
        if use_image:
            return "image"
        return "unconditional"

    def _predict_model_output_cfg(self, x_t, t, batch, branch_name: str, guidance_scale: float):
        preds = self.denoiser(x_t, t, batch)
        if branch_name == "unconditional" or guidance_scale == 1.0:
            return preds[branch_name]
        return preds["unconditional"] + guidance_scale * (preds[branch_name] - preds["unconditional"])

    @torch.no_grad()
    def sample_normalized(
        self,
        batch,
        guidance_scale: float = 1.2,
        use_text: bool = True,
        use_image: bool = True,
        clip_denoised: float | None = 1.0,
    ):
        device = next(self.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        b, n = batch["function_id"].shape
        mask = batch.get("mask", torch.ones(b, n, device=device)).to(device=device)
        x = torch.randn(b, n, self.latent_dim, device=device) * mask[..., None]
        branch_name = self._choose_branch(use_text=use_text, use_image=use_image)

        for t_value in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((b,), t_value, device=device, dtype=torch.long)
            model_output = self._predict_model_output_cfg(
                x,
                t,
                batch,
                branch_name=branch_name,
                guidance_scale=guidance_scale,
            )
            x0, eps = self._x0_and_eps_from_model_output(x, t, model_output)
            if clip_denoised is not None and clip_denoised > 0:
                x0 = x0.clamp(-float(clip_denoised), float(clip_denoised))
                if self.objective != "pred_x0":
                    eps = (x - torch.sqrt(self.alphas_cumprod[t_value]) * x0) / torch.sqrt(
                        torch.clamp(1.0 - self.alphas_cumprod[t_value], min=1e-12)
                    )

            alpha_bar = self.alphas_cumprod[t_value]
            alpha_bar_prev = self.alphas_cumprod[t_value - 1] if t_value > 0 else torch.ones_like(alpha_bar)
            beta = self.betas[t_value]
            alpha = 1.0 - beta
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
