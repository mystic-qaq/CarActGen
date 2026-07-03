from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from torch.utils.data import Dataset
from rich import print

from .functions import FUNCTION_TO_ID, load_function_map


class FunctionAwareGenSDFDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        train: bool | None,
        samples_per_mesh: int,
        pc_size: int,
        uniform_sample_ratio: float,
        mesh_info_dir: Path | None = None,
        sdf_subdir: str = "2_gensdf_dataset",
        uniform_sample_ratio_by_function: Dict[str, float] | None = None,
        sample_repeat_by_function: Dict[str, int] | None = None,
        point_cloud_surface_ratio: float = 0.0,
        point_cloud_surface_ratio_by_function: Dict[str, float] | None = None,
        point_cloud_surface_abs_percentile: float = 70.0,
        limit: int = -1,
        include_shape_ids: Iterable[str] | None = None,
        exclude_shape_ids: Iterable[str] | None = None,
    ):
        super().__init__()
        assert train is None, "Only support train=None, matching the original pipeline."

        dataset_dir = Path(dataset_dir)
        meta_path = dataset_dir / "meta.json"
        if meta_path.exists():
            json.loads(meta_path.read_text())

        include_shape_ids = set(include_shape_ids or [])
        exclude_shape_ids = set(exclude_shape_ids or [])

        def shape_id_from_file(path: Path) -> str:
            stem = path.stem.replace(".sdf", "")
            return stem.rsplit("_", 1)[0]

        files = sorted((dataset_dir / sdf_subdir).glob("*.npz"))
        if include_shape_ids:
            files = [path for path in files if shape_id_from_file(path) in include_shape_ids]
        if exclude_shape_ids:
            files = [path for path in files if shape_id_from_file(path) not in exclude_shape_ids]
        if limit is not None and int(limit) > 0:
            files = files[:int(limit)]
        if not files:
            raise FileNotFoundError(f"No SDF npz files found in {dataset_dir / sdf_subdir} after split filtering")
        mesh_info_dir = Path(mesh_info_dir) if mesh_info_dir else dataset_dir / "1_preprocessed_info"
        self.function_map = load_function_map(mesh_info_dir)
        self.default_function = (FUNCTION_TO_ID["static_part"], "static_part")

        sample_repeat_by_function = sample_repeat_by_function or {}
        expanded_files = []
        for file in files:
            stem = file.stem.replace(".sdf", "")
            _, label = self.function_map.get(stem, self.default_function)
            repeat = max(1, int(sample_repeat_by_function.get(label, 1)))
            expanded_files.extend([file] * repeat)

        random.shuffle(expanded_files)
        self.dataset_dir = expanded_files
        self.samples_per_mesh = int(samples_per_mesh)
        self.pc_size = int(pc_size)
        self.uniform_sample_ratio = float(uniform_sample_ratio)
        self.uniform_sample_ratio_by_function = uniform_sample_ratio_by_function or {}
        self.point_cloud_surface_ratio = float(point_cloud_surface_ratio)
        self.point_cloud_surface_ratio_by_function = point_cloud_surface_ratio_by_function or {}
        self.point_cloud_surface_abs_percentile = float(point_cloud_surface_abs_percentile)

        print("Len =", len(self.dataset_dir))

    def __len__(self):
        return len(self.dataset_dir)

    def select_point(self, point, sdf, n_point):
        half = int(n_point / 2)

        neg_idx = np.where(sdf < 0)[0]
        pos_idx = np.where(~(sdf < 0))[0]
        all_idx = np.arange(point.shape[0])
        assert all_idx.shape[0] > 0, "Empty point array"

        def take(indices, count):
            if count <= 0:
                return np.empty((0,), dtype=np.int64)
            if indices.shape[0] == 0:
                return np.empty((0,), dtype=np.int64)
            return np.random.choice(indices, size=count, replace=indices.shape[0] < count)

        neg_count = min(half, neg_idx.shape[0])
        pos_count = min(n_point - neg_count, pos_idx.shape[0])
        if neg_count + pos_count < n_point:
            neg_count = min(n_point - pos_count, neg_idx.shape[0])

        idx = np.concatenate([take(neg_idx, neg_count), take(pos_idx, pos_count)])
        if idx.shape[0] < n_point:
            idx = np.concatenate([idx, take(all_idx, n_point - idx.shape[0])])
        np.random.shuffle(idx)
        return point[idx], sdf[idx]

    def _function_for_file(self, file: Path) -> Tuple[int, str]:
        stem = file.stem.replace(".sdf", "")
        return self.function_map.get(stem, self.default_function)

    def _sample_points(self, points, count):
        if count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        points = np.asarray(points)
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        idx = np.random.choice(points.shape[0], size=count, replace=points.shape[0] < count)
        return points[idx].astype(np.float32)

    def _build_point_cloud(self, data, function_label: str):
        surface_ratio = self.point_cloud_surface_ratio_by_function.get(
            function_label,
            self.point_cloud_surface_ratio,
        )
        surface_ratio = min(max(float(surface_ratio), 0.0), 1.0)
        n_surface = int(round(self.pc_size * surface_ratio))
        n_on = self.pc_size - n_surface

        on_points = np.asarray(data["point_on"])
        if n_surface > 0:
            surface_points = np.asarray(data["point_surface"])
            surface_sdf = np.asarray(data["sdf_surface"])
            threshold = np.percentile(np.abs(surface_sdf), self.point_cloud_surface_abs_percentile)
            near_surface_points = surface_points[np.abs(surface_sdf) <= threshold]
        else:
            near_surface_points = np.empty((0, 3), dtype=np.float32)

        point_cloud = np.concatenate([
            self._sample_points(on_points, n_on),
            self._sample_points(near_surface_points, n_surface),
        ], axis=0)
        if point_cloud.shape[0] < self.pc_size:
            point_cloud = np.concatenate([
                point_cloud,
                self._sample_points(on_points, self.pc_size - point_cloud.shape[0]),
            ], axis=0)
        np.random.shuffle(point_cloud)
        return point_cloud.astype(np.float32)

    def __getitem__(self, index):
        file = self.dataset_dir[index]
        function_id, function_label = self._function_for_file(file)
        data = np.load(file.as_posix(), allow_pickle=True)

        ratio = self.uniform_sample_ratio_by_function.get(function_label, self.uniform_sample_ratio)
        n_uniform_point = int(self.samples_per_mesh * ratio)
        n_near_surface_point = self.samples_per_mesh - n_uniform_point

        uniform_point, uniform_sdf = self.select_point(data["point_uniform"], data["sdf_uniform"], n_uniform_point)
        surface_point, surface_sdf = self.select_point(data["point_surface"], data["sdf_surface"], n_near_surface_point)

        point_cloud = self._build_point_cloud(data, function_label)

        uniform_point = uniform_point.astype(np.float32)
        uniform_sdf = uniform_sdf.astype(np.float32)
        surface_point = surface_point.astype(np.float32)
        surface_sdf = surface_sdf.astype(np.float32)
        xyz = np.concatenate([uniform_point, surface_point])
        gt_sdf = np.concatenate([uniform_sdf, surface_sdf])
        idx = np.random.permutation(xyz.shape[0])

        return {
            "xyz": xyz[idx],
            "gt_sdf": gt_sdf[idx],
            "point_cloud": point_cloud,
            "function_id": np.array(function_id, dtype=np.int64),
            "function_label": function_label,
            "filename": file.stem,
        }
