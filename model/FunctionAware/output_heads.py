from __future__ import annotations

import torch
from torch import nn


class FunctionAwareOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, num_functions: int, film_scale: float = 0.1):
        super().__init__()
        self.num_functions = int(num_functions)
        self.film_scale = float(film_scale)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scale = nn.Embedding(self.num_functions, latent_dim)
        self.shift = nn.Embedding(self.num_functions, latent_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, hidden: torch.Tensor, function_id: torch.Tensor):
        function_id = function_id.to(hidden.device, dtype=torch.long).clamp(0, self.num_functions - 1)
        out = self.proj(hidden)
        gamma = torch.tanh(self.scale(function_id)) * self.film_scale
        beta = self.shift(function_id)
        return out * (1.0 + gamma) + beta
