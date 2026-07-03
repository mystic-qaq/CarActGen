from __future__ import annotations

import json
from collections import defaultdict
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


def _shape_stem(stem: str) -> str:
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem


def _part_sort_key(path: Path):
    tail = path.stem.split("_")[-1]
    return (0, int(tail)) if tail.isdigit() else (1, path.stem)


class AdaptiveObjectMultimodalDiffusionDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        text_shape=(16, 1024),
        image_shape=(32, 1408),
        max_parts: int = 16,
        num_functions: int = 8,
        cache: bool = True,
        normalize_latents: bool = True,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.text_shape = tuple(text_shape)
        self.image_shape = tuple(image_shape)
        self.max_parts = int(max_parts)
        self.num_functions = int(num_functions)
        self.normalize_latents = bool(normalize_latents)

        files = sorted(p for p in self.dataset_path.glob("*.npz") if p.name not in {"meta.npz", "latent_stats.npz", "function_latent_stats.npz"})
        if not files:
            raise FileNotFoundError(f"No latent npz files found in {self.dataset_path}")

        grouped = defaultdict(list)
        for file in files:
            grouped[_shape_stem(file.stem)].append(file)
        self.objects = [(shape_id, sorted(parts, key=_part_sort_key)[: self.max_parts]) for shape_id, parts in sorted(grouped.items())]
        self.objects = [item for item in self.objects if item[1]]
        if not self.objects:
            raise FileNotFoundError(f"No grouped objects found in {self.dataset_path}")

        meta_path = self.dataset_path / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.stats_path = self.dataset_path / "function_latent_stats.npz"
        self.function_mean, self.function_std = self._load_or_compute_stats(files)

        self.data = None
        if cache:
            self.data = [None] * len(self.objects)
            for idx in trange(len(self.objects), desc="Loading adaptive object diffusion data"):
                self.data[idx] = self._load_object(*self.objects[idx])

    def _load_or_compute_stats(self, files):
        if self.stats_path.exists():
            stats = np.load(self.stats_path, allow_pickle=True)
            return stats["mean"].astype(np.float32), np.maximum(stats["std"].astype(np.float32), 1e-6)

        all_latents = []
        by_function = [[] for _ in range(self.num_functions)]
        for file in files:
            data = np.load(file, allow_pickle=True)
            latent = data["latent_code"].astype(np.float32)
            function_id = int(data["function_id"]) if "function_id" in data.files else 0
            function_id = max(0, min(function_id, self.num_functions - 1))
            all_latents.append(latent)
            by_function[function_id].append(latent)

        global_stack = np.stack(all_latents, axis=0)
        global_mean = global_stack.mean(axis=0).astype(np.float32)
        global_std = np.maximum(global_stack.std(axis=0).astype(np.float32), 1e-6)
        means = np.repeat(global_mean[None], self.num_functions, axis=0)
        stds = np.repeat(global_std[None], self.num_functions, axis=0)
        counts = np.zeros(self.num_functions, dtype=np.int64)
        for function_id, values in enumerate(by_function):
            counts[function_id] = len(values)
            if values:
                stack = np.stack(values, axis=0)
                means[function_id] = stack.mean(axis=0).astype(np.float32)
                stds[function_id] = np.maximum(stack.std(axis=0).astype(np.float32), 1e-6)

        np.savez(self.stats_path, mean=means, std=stds, counts=counts)
        return means, stds

    def get_gensdf_ckpt_path(self):
        return self.meta["ckpt"]

    def __len__(self):
        return len(self.objects)

    def _normalize_latent(self, latent, function_id):
        latent = latent.astype(np.float32)
        if not self.normalize_latents:
            return latent
        function_id = max(0, min(int(function_id), self.num_functions - 1))
        return ((latent - self.function_mean[function_id]) / self.function_std[function_id]).astype(np.float32)

    def _load_part(self, file: Path):
        data = np.load(file, allow_pickle=True)
        function_id = int(data["function_id"]) if "function_id" in data.files else 0
        latent = self._normalize_latent(data["latent_code"], function_id)

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

        return latent, text, image, float(has_text), float(has_image), int(function_id), file.stem

    def _load_object(self, shape_id, files):
        latent = np.zeros((self.max_parts, self.function_mean.shape[-1]), dtype=np.float32)
        text = np.zeros((self.max_parts,) + self.text_shape, dtype=np.float32)
        image = np.zeros((self.max_parts,) + self.image_shape, dtype=np.float32)
        has_text = np.zeros((self.max_parts,), dtype=np.float32)
        has_image = np.zeros((self.max_parts,), dtype=np.float32)
        function_id = np.zeros((self.max_parts,), dtype=np.int64)
        mask = np.zeros((self.max_parts,), dtype=np.float32)
        stems = []

        for idx, file in enumerate(files[: self.max_parts]):
            z, t, im, ht, hi, fn, stem = self._load_part(file)
            latent[idx] = z
            text[idx] = t
            image[idx] = im
            has_text[idx] = ht
            has_image[idx] = hi
            function_id[idx] = fn
            mask[idx] = 1.0
            stems.append(stem)

        return {
            "latent_code": latent,
            "text": text,
            "image": image,
            "has_text": has_text,
            "has_image": has_image,
            "function_id": function_id,
            "mask": mask,
            "shape_id": shape_id,
            "part_stems": stems,
        }

    def __getitem__(self, index):
        if self.data is not None:
            return self.data[index]
        return self._load_object(*self.objects[index])
