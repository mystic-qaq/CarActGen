from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import Dataset
from tqdm import trange


def _first_existing_array(data, keys: Iterable[str]):
    for key in keys:
        if key in data.files:
            return data[key]
    return None


class AdaptiveMultimodalDiffusionDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        text_shape=(16, 1024),
        image_shape=(32, 1408),
        cache: bool = True,
        normalize_latents: bool = True,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.files = sorted(p for p in self.dataset_path.glob("*.npz") if p.name not in {"meta.npz", "latent_stats.npz"})
        if not self.files:
            raise FileNotFoundError(f"No latent npz files found in {self.dataset_path}")

        meta_path = self.dataset_path / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.text_shape = tuple(text_shape)
        self.image_shape = tuple(image_shape)
        self.normalize_latents = bool(normalize_latents)

        self.stats_path = Path(self.meta.get("latent_stats_path", self.dataset_path / "latent_stats.npz"))
        if not self.stats_path.is_absolute():
            self.stats_path = self.dataset_path / self.stats_path
        self.latent_mean, self.latent_std = self._load_or_compute_stats()

        self.data = None
        if cache:
            self.data = [None] * len(self.files)
            for idx in trange(len(self.files), desc="Loading adaptive multimodal diffusion data"):
                self.data[idx] = self._load_file(self.files[idx])

    def _load_or_compute_stats(self):
        if self.stats_path.exists():
            stats = np.load(self.stats_path, allow_pickle=True)
            return stats["mean"].astype(np.float32), np.maximum(stats["std"].astype(np.float32), 1e-6)

        latents = []
        for file in self.files:
            data = np.load(file, allow_pickle=True)
            latents.append(data["latent_code"].astype(np.float32))
        stacked = np.stack(latents, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = np.maximum(stacked.std(axis=0).astype(np.float32), 1e-6)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.stats_path, mean=mean, std=std)
        return mean, std

    def get_gensdf_ckpt_path(self):
        return self.meta["ckpt"]

    def __len__(self):
        return len(self.files)

    def _normalize_latent(self, latent):
        if not self.normalize_latents:
            return latent.astype(np.float32)
        return ((latent.astype(np.float32) - self.latent_mean) / self.latent_std).astype(np.float32)

    def _load_file(self, file: Path):
        data = np.load(file, allow_pickle=True)
        latent = self._normalize_latent(data["latent_code"])

        text = _first_existing_array(data, ["text", "text_embedding", "text_embed"])
        has_text = text is not None
        if text is None:
            text = np.zeros(self.text_shape, dtype=np.float32)
        text = text.astype(np.float32)

        image = _first_existing_array(data, ["image", "image_embedding", "image_embed", "image_feature", "images"])
        has_image = image is not None
        if image is None:
            image = np.zeros(self.image_shape, dtype=np.float32)
        image = image.astype(np.float32)
        if image.ndim == 1:
            image = image[None, :]

        return {
            "latent_code": latent,
            "text": text,
            "image": image,
            "has_text": np.array(float(has_text), dtype=np.float32),
            "has_image": np.array(float(has_image), dtype=np.float32),
            "function_id": data["function_id"].astype(np.int64) if "function_id" in data.files else np.array(0, dtype=np.int64),
            "function_label": str(data["function_label"].item()) if "function_label" in data.files else "",
            "filename": file.stem,
        }

    def __getitem__(self, index):
        if self.data is not None:
            return self.data[index]
        return self._load_file(self.files[index])
