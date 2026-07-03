from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PART_ORDER = [
    "body_shell",
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
]
WHEEL_ORDER = PART_ORDER[1:]
FUNCTION_TO_ID = {
    "static_root": 0,
    "rotation": 1,
    "static_part": 2,
    "translation": 3,
}


@dataclass
class LayoutArrays:
    shape_ids: list[str]
    latents: np.ndarray
    text: np.ndarray
    image: np.ndarray
    function_ids: np.ndarray
    target: np.ndarray
    body_center: np.ndarray
    body_size: np.ndarray
    part_center: np.ndarray
    part_size: np.ndarray
    wheel_pivots: np.ndarray


def safe_size(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(np.asarray(value, dtype=np.float32), eps)


def center_size_from_minmax(bbx: list[list[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbx = np.asarray(bbx, dtype=np.float32)
    center = (bbx[0] + bbx[1]) * 0.5
    size = safe_size(bbx[1] - bbx[0])
    return center.astype(np.float32), size.astype(np.float32)


def center_size_to_minmax(center: np.ndarray, size: np.ndarray) -> list[list[float]]:
    center = np.asarray(center, dtype=np.float32)
    size = safe_size(size)
    return [(center - 0.5 * size).tolist(), (center + 0.5 * size).tolist()]


def load_info_layout(info_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(info_path.read_text())
    by_name = {part["name"]: part for part in data["part"]}
    missing = [name for name in PART_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"{info_path.name} is missing required parts: {missing}")

    centers = []
    sizes = []
    for name in PART_ORDER:
        center, size = center_size_from_minmax(by_name[name]["bbx"])
        centers.append(center)
        sizes.append(size)
    centers = np.stack(centers).astype(np.float32)
    sizes = np.stack(sizes).astype(np.float32)
    body_center = centers[0]
    body_size = safe_size(sizes[0])
    pivots = np.asarray([by_name[name]["joint_data_origin"] for name in WHEEL_ORDER], dtype=np.float32)
    target = encode_layout_target(body_center, body_size, centers, sizes, pivots)
    return target, body_center, body_size, centers, sizes, pivots


def encode_layout_target(
    body_center: np.ndarray,
    body_size: np.ndarray,
    part_center: np.ndarray,
    part_size: np.ndarray,
    wheel_pivots: np.ndarray,
) -> np.ndarray:
    body_center = np.asarray(body_center, dtype=np.float32)
    body_size = safe_size(body_size)
    part_center = np.asarray(part_center, dtype=np.float32)
    part_size = safe_size(part_size)
    wheel_pivots = np.asarray(wheel_pivots, dtype=np.float32)

    chunks = [body_center, np.log(body_size)]
    wheel_centers = (part_center[1:] - body_center[None, :]) / body_size[None, :]
    wheel_sizes = np.log(part_size[1:] / body_size[None, :])
    pivot_rel = (wheel_pivots - body_center[None, :]) / body_size[None, :]
    chunks.extend([wheel_centers.reshape(-1), wheel_sizes.reshape(-1), pivot_rel.reshape(-1)])
    return np.concatenate(chunks).astype(np.float32)


def decode_layout_target(vector: np.ndarray, symmetrize: bool = False) -> dict[str, dict]:
    vector = np.asarray(vector, dtype=np.float32)
    body_center = vector[0:3]
    body_size = safe_size(np.exp(np.clip(vector[3:6], -6.0, 6.0)))
    cursor = 6
    wheel_center_rel = vector[cursor : cursor + 12].reshape(4, 3)
    cursor += 12
    wheel_size_rel = np.exp(np.clip(vector[cursor : cursor + 12].reshape(4, 3), -6.0, 6.0))
    cursor += 12
    pivot_rel = vector[cursor : cursor + 12].reshape(4, 3)

    wheel_centers = body_center[None, :] + wheel_center_rel * body_size[None, :]
    wheel_sizes = safe_size(wheel_size_rel * body_size[None, :])
    pivots = body_center[None, :] + pivot_rel * body_size[None, :]

    if symmetrize:
        wheel_centers, wheel_sizes, pivots = symmetrize_wheels(wheel_centers, wheel_sizes, pivots)

    layout = {
        "body_shell": {
            "center": body_center.tolist(),
            "size": body_size.tolist(),
            "bbx": [body_center.tolist(), body_size.tolist()],
            "joint_data_origin": [0.0, 0.0, 0.0],
            "joint_data_direction": [0.0, 0.0, 0.0],
        }
    }
    for idx, name in enumerate(WHEEL_ORDER):
        layout[name] = {
            "center": wheel_centers[idx].tolist(),
            "size": wheel_sizes[idx].tolist(),
            "bbx": [wheel_centers[idx].tolist(), wheel_sizes[idx].tolist()],
            "joint_data_origin": pivots[idx].tolist(),
            "joint_data_direction": [0.0, 1.0, 0.0],
        }
    return layout


def symmetrize_wheels(
    wheel_centers: np.ndarray,
    wheel_sizes: np.ndarray,
    pivots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(wheel_centers, dtype=np.float32).copy()
    sizes = np.asarray(wheel_sizes, dtype=np.float32).copy()
    anchors = np.asarray(pivots, dtype=np.float32).copy()
    for left, right in [(0, 1), (2, 3)]:
        for arr in [centers, anchors]:
            x = 0.5 * (arr[left, 0] + arr[right, 0])
            y_abs = 0.5 * (abs(arr[left, 1]) + abs(arr[right, 1]))
            z = 0.5 * (arr[left, 2] + arr[right, 2])
            arr[left] = [x, -y_abs, z]
            arr[right] = [x, y_abs, z]
        pair_size = 0.5 * (sizes[left] + sizes[right])
        sizes[left] = pair_size
        sizes[right] = pair_size
    return centers, safe_size(sizes), anchors


def pool_text(text: np.ndarray) -> np.ndarray:
    text = np.asarray(text, dtype=np.float32)
    if text.ndim == 1:
        return text
    return text.mean(axis=-2)


def pool_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 1:
        return np.concatenate([image, image], axis=-1)
    return np.concatenate([image.mean(axis=-2), image.max(axis=-2)], axis=-1).astype(np.float32)


def load_part_condition(condition_root: Path, shape_id: str, part_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    path = condition_root / f"{shape_id}_{part_idx}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing condition file: {path}")
    data = np.load(path, allow_pickle=True)
    latent = np.asarray(data["latent_code"], dtype=np.float32)
    text = pool_text(data["text"] if "text" in data.files else np.zeros((16, 1024), dtype=np.float32))
    image = pool_image(data["image"] if "image" in data.files else np.zeros((32, 768), dtype=np.float32))
    function_id = int(data["function_id"]) if "function_id" in data.files else 0
    return latent, text, image, function_id


def load_shape_condition(condition_root: Path, shape_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latents = []
    texts = []
    images = []
    function_ids = []
    for part_idx in range(1, len(PART_ORDER) + 1):
        latent, text, image, function_id = load_part_condition(condition_root, shape_id, part_idx)
        latents.append(latent)
        texts.append(text)
        images.append(image)
        function_ids.append(function_id)
    return (
        np.stack(latents).astype(np.float32),
        np.stack(texts).astype(np.float32),
        np.stack(images).astype(np.float32),
        np.asarray(function_ids, dtype=np.int64),
    )


def normalize_inputs(
    latents: torch.Tensor,
    text: torch.Tensor,
    image: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = (latents - stats["latent_mean"]) / stats["latent_std"]
    text = (text - stats["text_mean"]) / stats["text_std"]
    image = (image - stats["image_mean"]) / stats["image_std"]
    return latents, text, image


class LayoutNet(nn.Module):
    def __init__(
        self,
        latent_dim: int = 768,
        text_dim: int = 1024,
        image_dim: int = 1536,
        max_function_id: int = 8,
        hidden: int = 192,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.part_embedding = nn.Embedding(len(PART_ORDER), 16)
        self.function_embedding = nn.Embedding(max_function_id, 16)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        per_dim = 128 + 96 + 128 + 16 + 16
        self.part_encoder = nn.Sequential(
            nn.Linear(per_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * len(PART_ORDER) + hidden * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 42),
        )

    def forward(
        self,
        latents: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        function_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, part_count = latents.shape[:2]
        part_ids = torch.arange(part_count, device=latents.device)[None, :].expand(batch_size, part_count)
        function_ids = function_ids.clamp(min=0, max=self.function_embedding.num_embeddings - 1)
        encoded = torch.cat(
            [
                self.latent_encoder(latents),
                self.text_encoder(text),
                self.image_encoder(image),
                self.function_embedding(function_ids),
                self.part_embedding(part_ids),
            ],
            dim=-1,
        )
        part_feat = self.part_encoder(encoded)
        global_feat = torch.cat([part_feat.mean(dim=1), part_feat.max(dim=1).values], dim=-1)
        return self.head(torch.cat([part_feat.flatten(1), global_feat], dim=-1))


def checkpoint_to_device_stats(checkpoint: dict, device: torch.device) -> dict[str, torch.Tensor]:
    stats = {}
    for key in ["latent_mean", "latent_std", "text_mean", "text_std", "image_mean", "image_std", "target_mean", "target_std"]:
        value = checkpoint[key]
        stats[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return stats


def load_layout_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[LayoutNet, dict[str, torch.Tensor], dict]:
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = LayoutNet(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = checkpoint_to_device_stats(checkpoint, device)
    return model, stats, checkpoint


@torch.no_grad()
def predict_layout_vector(
    model: LayoutNet,
    stats: dict[str, torch.Tensor],
    latents: np.ndarray | torch.Tensor,
    text: np.ndarray | torch.Tensor,
    image: np.ndarray | torch.Tensor,
    function_ids: np.ndarray | torch.Tensor,
    device: torch.device | str,
) -> np.ndarray:
    device = torch.device(device)
    if not torch.is_tensor(latents):
        latents = torch.from_numpy(np.asarray(latents, dtype=np.float32))
    if not torch.is_tensor(text):
        text = torch.from_numpy(np.asarray(text, dtype=np.float32))
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
    if not torch.is_tensor(function_ids):
        function_ids = torch.from_numpy(np.asarray(function_ids, dtype=np.int64))
    latents = latents.to(device=device, dtype=torch.float32)
    text = text.to(device=device, dtype=torch.float32)
    image = image.to(device=device, dtype=torch.float32)
    function_ids = function_ids.to(device=device, dtype=torch.long)
    if latents.ndim == 2:
        latents = latents[None, :]
        text = text[None, :]
        image = image[None, :]
        function_ids = function_ids[None, :]
    latents, text, image = normalize_inputs(latents, text, image, stats)
    pred_std = model(latents, text, image, function_ids)
    pred = pred_std * stats["target_std"][None, :] + stats["target_mean"][None, :]
    return pred[0].detach().cpu().numpy().astype(np.float32)


def build_parts_from_layout(source_parts: dict, layout: dict[str, dict]) -> list[dict]:
    from experiments.fixed_car_template import CANONICAL_TEMPLATE

    parts = []
    for name in PART_ORDER:
        spec = copy.deepcopy(CANONICAL_TEMPLATE[name])
        spec["name"] = name
        spec["bbx"] = layout[name]["bbx"]
        spec["joint_data_origin"] = layout[name]["joint_data_origin"]
        spec["joint_data_direction"] = layout[name]["joint_data_direction"]
        spec["mesh"] = source_parts[name]["mesh"].copy()
        spec["source_mesh_path"] = source_parts[name]["mesh_path"].as_posix()
        parts.append(spec)
    return parts


def batch_from_sample_condition(batch: dict, latents: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if torch.is_tensor(latents):
        latents_np = latents.detach().cpu().numpy().astype(np.float32)
    else:
        latents_np = np.asarray(latents, dtype=np.float32)
    text = batch["text"][0, : len(PART_ORDER)].detach().cpu().numpy().astype(np.float32)
    image = batch["image"][0, : len(PART_ORDER)].detach().cpu().numpy().astype(np.float32)
    function_ids = batch["function_id"][0, : len(PART_ORDER)].detach().cpu().numpy().astype(np.int64)
    text_pool = np.stack([pool_text(item) for item in text]).astype(np.float32)
    image_pool = np.stack([pool_image(item) for item in image]).astype(np.float32)
    return latents_np, text_pool, image_pool, function_ids

