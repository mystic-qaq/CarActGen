from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.layout_net import (
    LayoutArrays,
    LayoutNet,
    PART_ORDER,
    WHEEL_ORDER,
    decode_layout_target,
    load_info_layout,
    load_shape_condition,
    normalize_inputs,
)


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_or_load_cache(args) -> dict[str, LayoutArrays]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or args.output_dir / "layout_net_cache.npz"
    if cache_path.exists() and not args.rebuild_cache:
        raw = np.load(cache_path, allow_pickle=True)
        out = {}
        for split in ["train", "val", "test"]:
            out[split] = LayoutArrays(
                shape_ids=[str(item) for item in raw[f"{split}_shape_ids"]],
                latents=raw[f"{split}_latents"].astype(np.float32),
                text=raw[f"{split}_text"].astype(np.float32),
                image=raw[f"{split}_image"].astype(np.float32),
                function_ids=raw[f"{split}_function_ids"].astype(np.int64),
                target=raw[f"{split}_target"].astype(np.float32),
                body_center=raw[f"{split}_body_center"].astype(np.float32),
                body_size=raw[f"{split}_body_size"].astype(np.float32),
                part_center=raw[f"{split}_part_center"].astype(np.float32),
                part_size=raw[f"{split}_part_size"].astype(np.float32),
                wheel_pivots=raw[f"{split}_wheel_pivots"].astype(np.float32),
            )
        return out

    split_json = json.loads(args.split_path.read_text())
    arrays = {}
    out = {}
    for split in ["train", "val", "test"]:
        shape_ids = list(split_json[split])
        latents = []
        text = []
        image = []
        function_ids = []
        target = []
        body_center = []
        body_size = []
        part_center = []
        part_size = []
        wheel_pivots = []
        for shape_id in shape_ids:
            z, txt, img, fid = load_shape_condition(args.condition_root, shape_id)
            tgt, bc, bs, pc, ps, piv = load_info_layout(args.info_root / f"{shape_id}.json")
            latents.append(z)
            text.append(txt)
            image.append(img)
            function_ids.append(fid)
            target.append(tgt)
            body_center.append(bc)
            body_size.append(bs)
            part_center.append(pc)
            part_size.append(ps)
            wheel_pivots.append(piv)
        data = LayoutArrays(
            shape_ids=shape_ids,
            latents=np.stack(latents).astype(np.float32),
            text=np.stack(text).astype(np.float32),
            image=np.stack(image).astype(np.float32),
            function_ids=np.stack(function_ids).astype(np.int64),
            target=np.stack(target).astype(np.float32),
            body_center=np.stack(body_center).astype(np.float32),
            body_size=np.stack(body_size).astype(np.float32),
            part_center=np.stack(part_center).astype(np.float32),
            part_size=np.stack(part_size).astype(np.float32),
            wheel_pivots=np.stack(wheel_pivots).astype(np.float32),
        )
        out[split] = data
        arrays[f"{split}_shape_ids"] = np.asarray(shape_ids, dtype=object)
        arrays[f"{split}_latents"] = data.latents
        arrays[f"{split}_text"] = data.text
        arrays[f"{split}_image"] = data.image
        arrays[f"{split}_function_ids"] = data.function_ids
        arrays[f"{split}_target"] = data.target
        arrays[f"{split}_body_center"] = data.body_center
        arrays[f"{split}_body_size"] = data.body_size
        arrays[f"{split}_part_center"] = data.part_center
        arrays[f"{split}_part_size"] = data.part_size
        arrays[f"{split}_wheel_pivots"] = data.wheel_pivots
    np.savez_compressed(cache_path, **arrays)
    return out


class LayoutDataset(Dataset):
    def __init__(
        self,
        data: LayoutArrays,
        stats_np: dict[str, np.ndarray],
        train: bool,
        latent_noise: float,
        condition_dropout: float,
    ):
        self.data = data
        self.stats_np = stats_np
        self.train = train
        self.latent_noise = latent_noise
        self.condition_dropout = condition_dropout

    def __len__(self):
        return len(self.data.shape_ids)

    def __getitem__(self, idx):
        latents = self.data.latents[idx].copy()
        text = self.data.text[idx].copy()
        image = self.data.image[idx].copy()
        if self.train:
            if self.latent_noise > 0:
                latents += np.random.normal(scale=self.latent_noise, size=latents.shape).astype(np.float32)
            if self.condition_dropout > 0:
                if random.random() < self.condition_dropout:
                    text[:] = 0.0
                if random.random() < self.condition_dropout:
                    image[:] = 0.0
        target = (self.data.target[idx] - self.stats_np["target_mean"]) / self.stats_np["target_std"]
        return {
            "latents": torch.from_numpy(latents.astype(np.float32)),
            "text": torch.from_numpy(text.astype(np.float32)),
            "image": torch.from_numpy(image.astype(np.float32)),
            "function_ids": torch.from_numpy(self.data.function_ids[idx].astype(np.int64)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_raw": torch.from_numpy(self.data.target[idx].astype(np.float32)),
        }


def build_stats(train_data: LayoutArrays) -> dict[str, np.ndarray]:
    stats = {
        "latent_mean": train_data.latents.reshape(-1, train_data.latents.shape[-1]).mean(axis=0, keepdims=True),
        "latent_std": train_data.latents.reshape(-1, train_data.latents.shape[-1]).std(axis=0, keepdims=True),
        "text_mean": train_data.text.reshape(-1, train_data.text.shape[-1]).mean(axis=0, keepdims=True),
        "text_std": train_data.text.reshape(-1, train_data.text.shape[-1]).std(axis=0, keepdims=True),
        "image_mean": train_data.image.reshape(-1, train_data.image.shape[-1]).mean(axis=0, keepdims=True),
        "image_std": train_data.image.reshape(-1, train_data.image.shape[-1]).std(axis=0, keepdims=True),
        "target_mean": train_data.target.mean(axis=0),
        "target_std": train_data.target.std(axis=0),
    }
    for key in ["latent_std", "text_std", "image_std"]:
        stats[key] = np.maximum(stats[key], 1e-5).astype(np.float32)
    stats["target_std"] = np.maximum(stats["target_std"], 1e-4).astype(np.float32)
    return {key: value.astype(np.float32) for key, value in stats.items()}


def stats_to_device(stats_np: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stats_np.items()}


def prior_loss(raw: torch.Tensor) -> torch.Tensor:
    wheel_center = raw[:, 6:18].reshape(-1, 4, 3)
    wheel_size_log = raw[:, 18:30].reshape(-1, 4, 3)
    pivot = raw[:, 30:42].reshape(-1, 4, 3)
    front_x = 0.5 * (wheel_center[:, 0, 0] + wheel_center[:, 1, 0])
    rear_x = 0.5 * (wheel_center[:, 2, 0] + wheel_center[:, 3, 0])
    front_pivot_x = 0.5 * (pivot[:, 0, 0] + pivot[:, 1, 0])
    rear_pivot_x = 0.5 * (pivot[:, 2, 0] + pivot[:, 3, 0])
    order = F.relu(front_x - rear_x + 0.10).mean() + F.relu(front_pivot_x - rear_pivot_x + 0.10).mean()
    y_sign = (
        F.relu(wheel_center[:, 0, 1] + 0.08).mean()
        + F.relu(0.08 - wheel_center[:, 1, 1]).mean()
        + F.relu(wheel_center[:, 2, 1] + 0.08).mean()
        + F.relu(0.08 - wheel_center[:, 3, 1]).mean()
    )
    size_plausible = F.relu(wheel_size_log.abs() - 3.0).mean()
    return order + 0.25 * y_sign + 0.05 * size_plausible


def batch_loss(pred_std: torch.Tensor, batch: dict, stats: dict[str, torch.Tensor], prior_weight: float) -> torch.Tensor:
    target_std = batch["target"].to(pred_std.device)
    loss_target = F.smooth_l1_loss(pred_std, target_std)
    pred_raw = pred_std * stats["target_std"][None, :] + stats["target_mean"][None, :]
    target_raw = batch["target_raw"].to(pred_std.device)
    loss_raw = F.smooth_l1_loss(pred_raw, target_raw)
    return loss_target + 0.25 * loss_raw + prior_weight * prior_loss(pred_raw)


def axis_iou(center_a: np.ndarray, size_a: np.ndarray, center_b: np.ndarray, size_b: np.ndarray) -> np.ndarray:
    min_a = center_a - 0.5 * size_a
    max_a = center_a + 0.5 * size_a
    min_b = center_b - 0.5 * size_b
    max_b = center_b + 0.5 * size_b
    inter = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    inter_vol = np.prod(inter, axis=-1)
    vol_a = np.prod(np.maximum(size_a, 1e-6), axis=-1)
    vol_b = np.prod(np.maximum(size_b, 1e-6), axis=-1)
    return inter_vol / np.maximum(vol_a + vol_b - inter_vol, 1e-6)


def physical_arrays_from_vectors(vectors: np.ndarray, symmetrize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    sizes = []
    pivots = []
    for vec in vectors:
        layout = decode_layout_target(vec, symmetrize=symmetrize)
        centers.append([layout[name]["center"] for name in PART_ORDER])
        sizes.append([layout[name]["size"] for name in PART_ORDER])
        pivots.append([layout[name]["joint_data_origin"] for name in WHEEL_ORDER])
    return np.asarray(centers, dtype=np.float32), np.asarray(sizes, dtype=np.float32), np.asarray(pivots, dtype=np.float32)


def metrics_for_vectors(pred: np.ndarray, data: LayoutArrays, symmetrize: bool = True) -> tuple[dict, list[dict]]:
    pred_center, pred_size, pred_pivot = physical_arrays_from_vectors(pred, symmetrize=symmetrize)
    center_err = np.linalg.norm(pred_center - data.part_center, axis=-1)
    size_err = np.abs(pred_size - data.part_size)
    pivot_err = np.linalg.norm(pred_pivot - data.wheel_pivots, axis=-1)
    iou = axis_iou(pred_center, pred_size, data.part_center, data.part_size)
    rows = []
    for idx, shape_id in enumerate(data.shape_ids):
        wheelbase_pred = pred_pivot[idx, 2:, 0].mean() - pred_pivot[idx, :2, 0].mean()
        wheelbase_gt = data.wheel_pivots[idx, 2:, 0].mean() - data.wheel_pivots[idx, :2, 0].mean()
        front_track_pred = pred_pivot[idx, 1, 1] - pred_pivot[idx, 0, 1]
        front_track_gt = data.wheel_pivots[idx, 1, 1] - data.wheel_pivots[idx, 0, 1]
        rear_track_pred = pred_pivot[idx, 3, 1] - pred_pivot[idx, 2, 1]
        rear_track_gt = data.wheel_pivots[idx, 3, 1] - data.wheel_pivots[idx, 2, 1]
        rows.append(
            {
                "shape_id": shape_id,
                "bbox_center_l2": float(center_err[idx].mean()),
                "body_center_l2": float(center_err[idx, 0]),
                "wheel_center_l2": float(center_err[idx, 1:].mean()),
                "bbox_size_mae": float(size_err[idx].mean()),
                "body_size_mae": float(size_err[idx, 0].mean()),
                "wheel_size_mae": float(size_err[idx, 1:].mean()),
                "bbox_iou": float(iou[idx].mean()),
                "body_iou": float(iou[idx, 0]),
                "wheel_iou": float(iou[idx, 1:].mean()),
                "pivot_l2": float(pivot_err[idx].mean()),
                "wheelbase_abs_err": float(abs(wheelbase_pred - wheelbase_gt)),
                "front_track_abs_err": float(abs(front_track_pred - front_track_gt)),
                "rear_track_abs_err": float(abs(rear_track_pred - rear_track_gt)),
            }
        )
    summary = {
        "n": len(data.shape_ids),
        "bbox_center_l2": float(center_err.mean()),
        "body_center_l2": float(center_err[:, 0].mean()),
        "wheel_center_l2": float(center_err[:, 1:].mean()),
        "bbox_size_mae": float(size_err.mean()),
        "body_size_mae": float(size_err[:, 0].mean()),
        "wheel_size_mae": float(size_err[:, 1:].mean()),
        "bbox_iou": float(iou.mean()),
        "body_iou": float(iou[:, 0].mean()),
        "wheel_iou": float(iou[:, 1:].mean()),
        "pivot_l2": float(pivot_err.mean()),
        "pivot_l2_p90": float(np.percentile(pivot_err, 90)),
        "wheelbase_abs_err": float(np.mean([row["wheelbase_abs_err"] for row in rows])),
        "front_track_abs_err": float(np.mean([row["front_track_abs_err"] for row in rows])),
        "rear_track_abs_err": float(np.mean([row["rear_track_abs_err"] for row in rows])),
    }
    return summary, rows


def write_summary(path: Path, summaries: dict[str, dict]):
    lines = [
        "# LayoutNet Clean Split Metrics",
        "",
        "| method | split | n | bbox center L2 | wheel center L2 | bbox size MAE | pivot L2 | wheelbase err | track err | bbox IoU | wheel IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summaries.items():
        method, split = key.split("/", 1)
        track = 0.5 * (row["front_track_abs_err"] + row["rear_track_abs_err"])
        lines.append(
            f"| {method} | {split} | {row['n']} | {row['bbox_center_l2']:.5f} | "
            f"{row['wheel_center_l2']:.5f} | {row['bbox_size_mae']:.5f} | {row['pivot_l2']:.5f} | "
            f"{row['wheelbase_abs_err']:.5f} | {track:.5f} | {row['bbox_iou']:.5f} | {row['wheel_iou']:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def predict(model: LayoutNet, loader: DataLoader, stats: dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for batch in loader:
        latents = batch["latents"].to(device)
        text = batch["text"].to(device)
        image = batch["image"].to(device)
        function_ids = batch["function_ids"].to(device)
        latents, text, image = normalize_inputs(latents, text, image, stats)
        pred_std = model(latents, text, image, function_ids)
        pred = pred_std * stats["target_std"][None, :] + stats["target_mean"][None, :]
        rows.append(pred.detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def train(args):
    seed_all(args.seed)
    splits = build_or_load_cache(args)
    stats_np = build_stats(splits["train"])
    stats_t = stats_to_device(stats_np, torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"))
    device = next(iter(stats_t.values())).device

    run_dir = args.output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_layout_net_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    np.savez(run_dir / "normalization.npz", **stats_np)

    datasets = {
        split: LayoutDataset(
            data,
            stats_np,
            train=split == "train",
            latent_noise=args.latent_noise,
            condition_dropout=args.condition_dropout,
        )
        for split, data in splits.items()
    }
    eval_datasets = {
        split: LayoutDataset(
            data,
            stats_np,
            train=False,
            latent_noise=0.0,
            condition_dropout=0.0,
        )
        for split, data in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    eval_loaders = {
        split: DataLoader(eval_datasets[split], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for split in ["train", "val", "test"]
    }

    model_config = {
        "latent_dim": splits["train"].latents.shape[-1],
        "text_dim": splits["train"].text.shape[-1],
        "image_dim": splits["train"].image.shape[-1],
        "max_function_id": max(8, int(splits["train"].function_ids.max()) + 2),
        "hidden": args.hidden,
        "dropout": args.dropout,
    }
    model = LayoutNet(**model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.04)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            latents = batch["latents"].to(device)
            text = batch["text"].to(device)
            image = batch["image"].to(device)
            function_ids = batch["function_ids"].to(device)
            latents, text, image = normalize_inputs(latents, text, image, stats_t)
            pred = model(latents, text, image, function_ids)
            loss = batch_loss(pred, batch, stats_t, args.prior_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_pred = predict(model, loaders["val"], stats_t, device)
            val_summary, _ = metrics_for_vectors(val_pred, splits["val"], symmetrize=True)
            train_loss = float(np.mean(losses))
            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_summary.items() if k != "n"}}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            val_metric = val_summary["pivot_l2"] + 0.5 * val_summary["wheel_center_l2"] + 0.1 * val_summary["bbox_size_mae"]
            if val_metric < best_val:
                best_val = val_metric
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model_config,
                        "epoch": epoch,
                        "best_val_metric": best_val,
                        **stats_np,
                    },
                    run_dir / "best.pt",
                )
            else:
                bad_epochs += args.eval_every
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])

    summaries = {}
    rows = []
    train_mean = np.broadcast_to(stats_np["target_mean"][None, :], splits["train"].target.shape)
    canonical = canonical_layout_vector()
    for split, data in splits.items():
        pred = predict(model, eval_loaders[split], stats_t, device)
        mean_pred = np.broadcast_to(stats_np["target_mean"][None, :], data.target.shape)
        canonical_pred = np.broadcast_to(canonical[None, :], data.target.shape)
        for method, values in [
            ("train_mean_layout", mean_pred),
            ("canonical_layout", canonical_pred),
            ("layout_net", pred),
        ]:
            summary, per_shape = metrics_for_vectors(values, data, symmetrize=True)
            summaries[f"{method}/{split}"] = summary
            for row in per_shape:
                rows.append({"method": method, "split": split, **row})

    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "per_shape_metrics.csv", rows)
    (run_dir / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_summary(run_dir / "summary.md", summaries)
    (run_dir / "DONE").write_text(f"best_epoch={best_epoch}\nbest_val_metric={best_val:.8f}\n")
    (args.output_dir / "latest_layout_net.txt").write_text(run_dir.as_posix() + "\n")
    print(run_dir.as_posix())


def canonical_layout_vector() -> np.ndarray:
    from experiments.fixed_car_template import CANONICAL_TEMPLATE
    from experiments.layout_net import encode_layout_target

    centers = []
    sizes = []
    pivots = []
    for name in PART_ORDER:
        center, size = CANONICAL_TEMPLATE[name]["bbx"]
        centers.append(center)
        sizes.append(size)
    for name in WHEEL_ORDER:
        pivots.append(CANONICAL_TEMPLATE[name]["joint_data_origin"])
    centers = np.asarray(centers, dtype=np.float32)
    sizes = np.asarray(sizes, dtype=np.float32)
    pivots = np.asarray(pivots, dtype=np.float32)
    return encode_layout_target(centers[0], sizes[0], centers, sizes, pivots)


def main():
    parser = argparse.ArgumentParser(description="Train a clean-split LayoutNet for fixed-schema articulated cars.")
    parser.add_argument("--split_path", type=Path, default=env_path("CARACTGEN_SPLIT_PATH", REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json"))
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument(
        "--condition_root",
        type=Path,
        default=env_path("CARACTGEN_CONDITION_ROOT", env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2"),
    )
    parser.add_argument("--output_dir", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_layout_net")
    parser.add_argument("--cache_path", type=Path)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--prior_weight", type=float, default=0.02)
    parser.add_argument("--latent_noise", type=float, default=0.02)
    parser.add_argument("--condition_dropout", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--patience", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
