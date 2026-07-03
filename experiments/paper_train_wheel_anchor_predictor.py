from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PART_ORDER = [
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))

CANONICAL_BODY_BBX = np.asarray([[1.45, 0.0, 0.48], [4.65, 1.98, 1.22]], dtype=np.float32)
CANONICAL_WHEEL_ORIGINS = np.asarray(
    [
        [0.01, -0.74, 0.0],
        [0.01, 0.74, 0.0],
        [2.75, -0.74, 0.0],
        [2.75, 0.74, 0.0],
    ],
    dtype=np.float32,
)


@dataclass
class SplitData:
    shape_ids: list[str]
    point_cloud: np.ndarray
    bbox_features: np.ndarray
    body_center: np.ndarray
    body_size: np.ndarray
    target_params: np.ndarray
    target_pivots: np.ndarray


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_size(size: np.ndarray) -> np.ndarray:
    return np.maximum(size.astype(np.float32), 1e-6)


def center_size_from_bbx(bbx: list[list[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbx = np.asarray(bbx, dtype=np.float32)
    center = (bbx[0] + bbx[1]) * 0.5
    size = safe_size(bbx[1] - bbx[0])
    return center, size


def pivots_to_params(pivots: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    norm = (pivots.astype(np.float32) - center[None, :]) / safe_size(size)[None, :]
    front = 0.5 * (norm[0] + norm[1])
    rear = 0.5 * (norm[2] + norm[3])
    front_y = 0.5 * (abs(norm[0, 1]) + abs(norm[1, 1]))
    rear_y = 0.5 * (abs(norm[2, 1]) + abs(norm[3, 1]))
    return np.asarray([front[0], front_y, front[2], rear[0], rear_y, rear[2]], dtype=np.float32)


def params_to_normalized_pivots(params: torch.Tensor) -> torch.Tensor:
    fx, fy, fz, rx, ry, rz = params.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([fx, -fy, fz], dim=-1),
            torch.stack([fx, fy, fz], dim=-1),
            torch.stack([rx, -ry, rz], dim=-1),
            torch.stack([rx, ry, rz], dim=-1),
        ],
        dim=-2,
    )


def params_to_pivots_np(params: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float32)
    norm = np.asarray(
        [
            [p[0], -p[1], p[2]],
            [p[0], p[1], p[2]],
            [p[3], -p[4], p[5]],
            [p[3], p[4], p[5]],
        ],
        dtype=np.float32,
    )
    return center[None, :] + norm * safe_size(size)[None, :]


def canonical_params() -> np.ndarray:
    center, size = center_size_from_bbx(CANONICAL_BODY_BBX)
    return pivots_to_params(CANONICAL_WHEEL_ORIGINS, center, size)


def load_shape_record(
    shape_id: str,
    info_root: Path,
    sdf_root: Path,
    pc_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    info_path = info_root / f"{shape_id}.json"
    data = json.loads(info_path.read_text())
    by_name = {part["name"]: part for part in data["part"]}
    body = by_name["body_shell"]
    center, size = center_size_from_bbx(body["bbx"])
    pivots = np.asarray([by_name[name]["joint_data_origin"] for name in PART_ORDER], dtype=np.float32)
    target_params = pivots_to_params(pivots, center, size)

    sdf_path = sdf_root / f"{shape_id}_1.sdf.npz"
    sdf = np.load(sdf_path, allow_pickle=True)
    points = np.asarray(sdf["point_on"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(points) >= pc_size:
        idx = rng.choice(len(points), size=pc_size, replace=False)
    else:
        idx = rng.choice(len(points), size=pc_size, replace=True)
    point_cloud = points[idx]
    pc_center = point_cloud.mean(axis=0, keepdims=True)
    pc_scale = np.maximum(np.percentile(np.abs(point_cloud - pc_center), 95), 1e-6)
    point_cloud = (point_cloud - pc_center) / pc_scale

    length, width, height = size.tolist()
    bbox_features = np.asarray(
        [
            length,
            width,
            height,
            width / max(length, 1e-6),
            height / max(length, 1e-6),
            height / max(width, 1e-6),
            math.log(max(length, 1e-6)),
            math.log(max(width, 1e-6)),
            math.log(max(height, 1e-6)),
            float(np.prod(size)),
        ],
        dtype=np.float32,
    )
    return point_cloud.astype(np.float32), bbox_features, center, size, target_params, pivots


def build_or_load_cache(args) -> dict[str, SplitData]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or args.output_dir / f"wheel_anchor_cache_pc{args.pc_size}.npz"
    if cache_path.exists() and not args.rebuild_cache:
        raw = np.load(cache_path, allow_pickle=True)
        splits = {}
        for split in ["train", "val", "test"]:
            splits[split] = SplitData(
                shape_ids=list(raw[f"{split}_shape_ids"]),
                point_cloud=raw[f"{split}_point_cloud"].astype(np.float32),
                bbox_features=raw[f"{split}_bbox_features"].astype(np.float32),
                body_center=raw[f"{split}_body_center"].astype(np.float32),
                body_size=raw[f"{split}_body_size"].astype(np.float32),
                target_params=raw[f"{split}_target_params"].astype(np.float32),
                target_pivots=raw[f"{split}_target_pivots"].astype(np.float32),
            )
        return splits

    split_json = json.loads(args.split_path.read_text())
    arrays = {}
    splits = {}
    for split in ["train", "val", "test"]:
        records = [
            load_shape_record(shape_id, args.info_root, args.sdf_root, args.pc_size, args.seed + i * 17)
            for i, shape_id in enumerate(split_json[split])
        ]
        point_cloud, bbox_features, centers, sizes, target_params, pivots = zip(*records)
        split_data = SplitData(
            shape_ids=list(split_json[split]),
            point_cloud=np.stack(point_cloud),
            bbox_features=np.stack(bbox_features),
            body_center=np.stack(centers),
            body_size=np.stack(sizes),
            target_params=np.stack(target_params),
            target_pivots=np.stack(pivots),
        )
        splits[split] = split_data
        arrays[f"{split}_shape_ids"] = np.asarray(split_data.shape_ids, dtype=object)
        arrays[f"{split}_point_cloud"] = split_data.point_cloud
        arrays[f"{split}_bbox_features"] = split_data.bbox_features
        arrays[f"{split}_body_center"] = split_data.body_center
        arrays[f"{split}_body_size"] = split_data.body_size
        arrays[f"{split}_target_params"] = split_data.target_params
        arrays[f"{split}_target_pivots"] = split_data.target_pivots
    np.savez_compressed(cache_path, **arrays)
    return splits


class WheelAnchorDataset(Dataset):
    def __init__(
        self,
        data: SplitData,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        train: bool,
        jitter_std: float,
        point_dropout: float,
    ):
        self.data = data
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = np.maximum(feature_std.astype(np.float32), 1e-6)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = np.maximum(target_std.astype(np.float32), 1e-6)
        self.train = train
        self.jitter_std = jitter_std
        self.point_dropout = point_dropout

    def __len__(self):
        return len(self.data.shape_ids)

    def __getitem__(self, idx):
        pc = self.data.point_cloud[idx].copy()
        if self.train:
            if self.point_dropout > 0:
                keep = np.random.random(len(pc)) >= self.point_dropout
                if keep.any():
                    kept = pc[keep]
                    refill = np.random.choice(len(kept), size=len(pc), replace=True)
                    pc = kept[refill]
            if self.jitter_std > 0:
                pc += np.random.normal(scale=self.jitter_std, size=pc.shape).astype(np.float32)
        features = (self.data.bbox_features[idx] - self.feature_mean) / self.feature_std
        target = (self.data.target_params[idx] - self.target_mean) / self.target_std
        return {
            "point_cloud": torch.from_numpy(pc),
            "features": torch.from_numpy(features.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_params": torch.from_numpy(self.data.target_params[idx]),
            "body_center": torch.from_numpy(self.data.body_center[idx]),
            "body_size": torch.from_numpy(self.data.body_size[idx]),
        }


class BBoxMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 6),
        )

    def forward(self, point_cloud: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PointNetAnchor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 192):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 2 + 64, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 6),
        )

    def forward(self, point_cloud: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = point_cloud.transpose(1, 2).contiguous()
        feat = self.point_mlp(x)
        pooled = torch.cat([feat.max(dim=-1).values, feat.mean(dim=-1)], dim=-1)
        global_feat = self.feature_mlp(features)
        return self.head(torch.cat([pooled, global_feat], dim=-1))


def make_model(name: str, feature_dim: int) -> nn.Module:
    if name == "bbox_mlp":
        return BBoxMLP(feature_dim)
    if name == "pointnet_anchor":
        return PointNetAnchor(feature_dim)
    raise ValueError(f"unknown model: {name}")


def prior_loss(params: torch.Tensor) -> torch.Tensor:
    fx, fy, _fz, rx, ry, _rz = params.unbind(dim=-1)
    order = F.relu(fx - rx + 0.20).mean()
    positive_track = F.relu(0.15 - fy).mean() + F.relu(0.15 - ry).mean()
    plausible_track = F.relu(fy - 0.55).mean() + F.relu(ry - 0.55).mean()
    return order + 0.5 * positive_track + 0.5 * plausible_track


def unstandardize(std_pred: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return std_pred * target_std[None, :] + target_mean[None, :]


def batch_loss(pred_std: torch.Tensor, batch: dict, target_mean: torch.Tensor, target_std: torch.Tensor, prior_weight: float):
    target_std_values = batch["target"].to(pred_std.device)
    target_params = batch["target_params"].to(pred_std.device)
    pred_params = unstandardize(pred_std, target_mean, target_std)
    loss_param = F.smooth_l1_loss(pred_std, target_std_values)
    pred_norm = params_to_normalized_pivots(pred_params)
    target_norm = params_to_normalized_pivots(target_params)
    loss_pivot = F.smooth_l1_loss(pred_norm, target_norm)
    loss_prior = prior_loss(pred_params)
    return loss_param + 0.5 * loss_pivot + prior_weight * loss_prior


def metrics_for_params(pred_params: np.ndarray, data: SplitData) -> tuple[dict, list[dict]]:
    rows = []
    errors = []
    param_errors = np.abs(pred_params - data.target_params)
    for i, shape_id in enumerate(data.shape_ids):
        pred_pivots = params_to_pivots_np(pred_params[i], data.body_center[i], data.body_size[i])
        target_pivots = data.target_pivots[i]
        diff = pred_pivots - target_pivots
        per_wheel_l2 = np.linalg.norm(diff, axis=-1)
        errors.append(per_wheel_l2)
        row = {
            "shape_id": shape_id,
            "mean_l2": float(per_wheel_l2.mean()),
            "max_l2": float(per_wheel_l2.max()),
            "x_mae": float(np.abs(diff[:, 0]).mean()),
            "y_mae": float(np.abs(diff[:, 1]).mean()),
            "z_mae": float(np.abs(diff[:, 2]).mean()),
            "front_l2": float(per_wheel_l2[:2].mean()),
            "rear_l2": float(per_wheel_l2[2:].mean()),
            "wheelbase_abs_err": float(
                abs((pred_pivots[2:, 0].mean() - pred_pivots[:2, 0].mean()) - (target_pivots[2:, 0].mean() - target_pivots[:2, 0].mean()))
            ),
            "front_track_abs_err": float(
                abs((pred_pivots[1, 1] - pred_pivots[0, 1]) - (target_pivots[1, 1] - target_pivots[0, 1]))
            ),
            "rear_track_abs_err": float(
                abs((pred_pivots[3, 1] - pred_pivots[2, 1]) - (target_pivots[3, 1] - target_pivots[2, 1]))
            ),
        }
        rows.append(row)
    E = np.asarray(errors)
    summary = {
        "n": len(data.shape_ids),
        "pivot_l2_mean": float(E.mean()),
        "pivot_l2_median": float(np.median(E)),
        "pivot_l2_p90": float(np.percentile(E, 90)),
        "pivot_l2_max": float(E.max()),
        "param_mae": float(param_errors.mean()),
        "front_x_param_mae": float(param_errors[:, 0].mean()),
        "front_y_param_mae": float(param_errors[:, 1].mean()),
        "front_z_param_mae": float(param_errors[:, 2].mean()),
        "rear_x_param_mae": float(param_errors[:, 3].mean()),
        "rear_y_param_mae": float(param_errors[:, 4].mean()),
        "rear_z_param_mae": float(param_errors[:, 5].mean()),
        "x_mae": float(np.mean([row["x_mae"] for row in rows])),
        "y_mae": float(np.mean([row["y_mae"] for row in rows])),
        "z_mae": float(np.mean([row["z_mae"] for row in rows])),
        "wheelbase_abs_err": float(np.mean([row["wheelbase_abs_err"] for row in rows])),
        "front_track_abs_err": float(np.mean([row["front_track_abs_err"] for row in rows])),
        "rear_track_abs_err": float(np.mean([row["rear_track_abs_err"] for row in rows])),
    }
    return summary, rows


@torch.no_grad()
def predict_model(model, loader, target_mean, target_std, device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        pc = batch["point_cloud"].to(device)
        features = batch["features"].to(device)
        pred_std = model(pc, features)
        pred_params = unstandardize(pred_std, target_mean, target_std)
        preds.append(pred_params.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, model_name: str, summaries: dict[str, dict]):
    lines = [
        f"# Wheel Anchor Prediction ({model_name})",
        "",
        "| method | split | n | pivot L2 mean | pivot L2 median | pivot L2 p90 | x MAE | y MAE | z MAE | wheelbase err | front track err | rear track err | param MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summaries.items():
        method, split = key.split("/", 1)
        lines.append(
            f"| {method} | {split} | {row['n']} | {row['pivot_l2_mean']:.5f} | "
            f"{row['pivot_l2_median']:.5f} | {row['pivot_l2_p90']:.5f} | "
            f"{row['x_mae']:.5f} | {row['y_mae']:.5f} | {row['z_mae']:.5f} | "
            f"{row['wheelbase_abs_err']:.5f} | {row['front_track_abs_err']:.5f} | "
            f"{row['rear_track_abs_err']:.5f} | {row['param_mae']:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def train(args):
    seed_all(args.seed)
    splits = build_or_load_cache(args)
    train_data = splits["train"]
    feature_mean = train_data.bbox_features.mean(axis=0)
    feature_std = train_data.bbox_features.std(axis=0)
    target_mean = train_data.target_params.mean(axis=0)
    target_std = train_data.target_params.std(axis=0)
    target_std = np.maximum(target_std, 1e-4)

    run_dir = args.output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    np.savez(
        run_dir / "normalization.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        canonical_params=canonical_params(),
    )

    datasets = {
        split: WheelAnchorDataset(
            data,
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            train=split == "train",
            jitter_std=args.jitter_std,
            point_dropout=args.point_dropout,
        )
        for split, data in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_model(args.model, train_data.bbox_features.shape[1]).to(device)
    target_mean_t = torch.from_numpy(target_mean.astype(np.float32)).to(device)
    target_std_t = torch.from_numpy(target_std.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["point_cloud"].to(device), batch["features"].to(device))
            loss = batch_loss(pred, batch, target_mean_t, target_std_t, args.prior_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_pred = predict_model(model, loaders["val"], target_mean_t, target_std_t, device)
            val_summary, _ = metrics_for_params(val_pred, splits["val"])
            train_loss = float(np.mean(losses))
            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_summary.items() if k != "n"}}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            val_metric = val_summary["pivot_l2_mean"]
            if val_metric < best_val:
                best_val = val_metric
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_name": args.model,
                        "epoch": epoch,
                        "best_val": best_val,
                        "feature_mean": feature_mean,
                        "feature_std": feature_std,
                        "target_mean": target_mean,
                        "target_std": target_std,
                    },
                    run_dir / "best.pt",
                )
            else:
                bad_epochs += args.eval_every
            if bad_epochs >= args.patience:
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    summaries = {}
    all_rows = []
    train_mean = np.broadcast_to(target_mean[None, :], splits["test"].target_params.shape)
    canonical = canonical_params()
    for split, data in splits.items():
        template_mean = np.broadcast_to(target_mean[None, :], data.target_params.shape)
        template_canon = np.broadcast_to(canonical[None, :], data.target_params.shape)
        pred = predict_model(model, loaders[split], target_mean_t, target_std_t, device)
        for method, params in [
            ("train_mean_template", template_mean),
            ("canonical_template", template_canon),
            (args.model, pred),
        ]:
            summary, rows = metrics_for_params(params, data)
            summaries[f"{method}/{split}"] = summary
            for row in rows:
                row = dict(row)
                row["method"] = method
                row["split"] = split
                all_rows.append(row)

    write_csv(run_dir / "per_shape_metrics.csv", all_rows)
    write_csv(run_dir / "history.csv", history)
    write_summary(run_dir / "summary.md", args.model, summaries)
    (run_dir / "DONE").write_text(f"best_epoch={best_epoch}\nbest_val={best_val:.8f}\n")
    latest = args.output_dir / f"latest_{args.model}.txt"
    latest.write_text(run_dir.as_posix() + "\n")
    print(run_dir.as_posix())


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight wheel-anchor predictor and compare with template assembly.")
    parser.add_argument("--model", choices=["bbox_mlp", "pointnet_anchor"], default="pointnet_anchor")
    parser.add_argument("--split_path", type=Path, default=env_path("CARACTGEN_SPLIT_PATH", REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json"))
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--sdf_root", type=Path, default=env_path("CARACTGEN_SDF_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "2_gensdf_dataset_adaptive"))
    parser.add_argument("--output_dir", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_wheel_anchor")
    parser.add_argument("--cache_path", type=Path)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--pc_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prior_weight", type=float, default=0.02)
    parser.add_argument("--jitter_std", type=float, default=0.01)
    parser.add_argument("--point_dropout", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
