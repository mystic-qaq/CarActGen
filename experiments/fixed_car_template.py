import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from utils import HighPrecisionJsonEncoder


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))


PART_ORDER = [
    "body_shell",
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
]

CANONICAL_TEMPLATE = {
    "body_shell": {
        "dfn": 1,
        "dfn_fa": 0,
        "bbx": [[1.45, 0.0, 0.48], [4.65, 1.98, 1.22]],
        "joint_data_origin": [0.0, 0.0, 0.0],
        "joint_data_direction": [0.0, 0.0, 0.0],
        "limit": [0.0, 0.0, 0.0, 0.0],
    },
    "wheel_front_left": {
        "dfn": 2,
        "dfn_fa": 1,
        "bbx": [[0.01, -0.74, 0.0], [0.64, 0.22, 0.64]],
        "joint_data_origin": [0.01, -0.74, 0.0],
        "joint_data_direction": [0.0, 1.0, 0.0],
        "limit": [0.0, 0.0, 0.0, 2.0 * math.pi],
    },
    "wheel_front_right": {
        "dfn": 3,
        "dfn_fa": 1,
        "bbx": [[0.01, 0.74, 0.0], [0.64, 0.22, 0.64]],
        "joint_data_origin": [0.01, 0.74, 0.0],
        "joint_data_direction": [0.0, 1.0, 0.0],
        "limit": [0.0, 0.0, 0.0, 2.0 * math.pi],
    },
    "wheel_rear_left": {
        "dfn": 4,
        "dfn_fa": 1,
        "bbx": [[2.75, -0.74, 0.0], [0.64, 0.22, 0.64]],
        "joint_data_origin": [2.75, -0.74, 0.0],
        "joint_data_direction": [0.0, 1.0, 0.0],
        "limit": [0.0, 0.0, 0.0, 2.0 * math.pi],
    },
    "wheel_rear_right": {
        "dfn": 5,
        "dfn_fa": 1,
        "bbx": [[2.75, 0.74, 0.0], [0.64, 0.22, 0.64]],
        "joint_data_origin": [2.75, 0.74, 0.0],
        "joint_data_direction": [0.0, 1.0, 0.0],
        "limit": [0.0, 0.0, 0.0, 2.0 * math.pi],
    },
}

COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]


def center_size_from_minmax(bbx):
    mins = np.asarray(bbx[0], dtype=float)
    maxs = np.asarray(bbx[1], dtype=float)
    center = ((mins + maxs) / 2.0).tolist()
    size = (maxs - mins).tolist()
    return [center, size]


def load_source_parts(shape_id: str, info_root: Path, mesh_root: Path):
    info_path = info_root / f"{shape_id}.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing source info: {info_path}")

    source = json.loads(info_path.read_text())
    by_name = {part["name"]: part for part in source["part"]}
    missing = [name for name in PART_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"{shape_id} is missing required parts: {missing}")

    parts = {}
    for name in PART_ORDER:
        source_part = by_name[name]
        mesh_path = mesh_root / source_part["mesh"]
        if not mesh_path.exists():
            raise FileNotFoundError(f"missing mesh: {mesh_path}")
        parts[name] = {
            "source": source_part,
            "mesh": trimesh.load_mesh(mesh_path.as_posix()),
            "mesh_path": mesh_path,
        }
    return source, parts


def build_parts(source_parts, template_mode: str):
    parts = []
    for name in PART_ORDER:
        src = source_parts[name]["source"]
        if template_mode == "source-bbox":
            spec = {
                "dfn": CANONICAL_TEMPLATE[name]["dfn"],
                "dfn_fa": CANONICAL_TEMPLATE[name]["dfn_fa"],
                "bbx": center_size_from_minmax(src["bbx"]),
                "joint_data_origin": src["joint_data_origin"],
                "joint_data_direction": CANONICAL_TEMPLATE[name]["joint_data_direction"],
                "limit": CANONICAL_TEMPLATE[name]["limit"],
            }
        else:
            spec = CANONICAL_TEMPLATE[name]

        part = copy.deepcopy(spec)
        part["name"] = name
        part["mesh"] = source_parts[name]["mesh"].copy()
        part["source_mesh_path"] = source_parts[name]["mesh_path"].as_posix()
        parts.append(part)
    return parts


def calc_linear_value(left, right, ratio):
    return left + (right - left) * ratio


def produce_translate_matrix(direction, distance):
    direction = np.asarray(direction, dtype=float)
    matrix = np.eye(4)
    matrix[:3, 3] = direction * distance
    return matrix


def produce_rotate_matrix(direction, angle):
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(4)
    direction = direction / norm
    kx, ky, kz = direction
    skew = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    rot = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    return matrix


def produce_rotate_around_line_matrix(origin, direction, angle):
    origin = np.asarray(origin, dtype=float)
    return (
        produce_translate_matrix(origin, 1.0)
        @ produce_rotate_matrix(direction, angle)
        @ produce_translate_matrix(-origin, 1.0)
    )


def fit_mesh_to_bbx(mesh: trimesh.Trimesh, bbx):
    fitted = mesh.copy()
    min_bound, max_bound = fitted.bounds
    center = np.asarray(bbx[0], dtype=float)
    size = np.asarray(bbx[1], dtype=float)

    target_min = center - size / 2.0
    target_max = center + size / 2.0

    max_bound = np.array(max_bound, dtype=float, copy=True)
    min_bound = np.array(min_bound, dtype=float, copy=True)
    max_bound[(max_bound - min_bound) < 1e-5] += 1e-3
    target_max[(target_max - target_min) < 1e-5] += 1e-3

    fitted.vertices = target_min + (target_max - target_min) * (
        (fitted.vertices - min_bound) / (max_bound - min_bound)
    )
    return fitted


def apply_pose(parts, ratio: float):
    by_dfn = {part["dfn"]: copy.deepcopy(part) for part in parts}
    for part in by_dfn.values():
        part["children"] = []

    for part in by_dfn.values():
        if part["dfn_fa"] in by_dfn:
            by_dfn[part["dfn_fa"]]["children"].append(part["dfn"])

    for part in by_dfn.values():
        if part["dfn_fa"] == 0:
            part["transform"] = np.eye(4)
            continue
        slide_distance = calc_linear_value(*part["limit"][:2], ratio)
        angle = calc_linear_value(*part["limit"][2:], ratio)
        part["transform"] = (
            produce_rotate_around_line_matrix(part["joint_data_origin"], part["joint_data_direction"], angle)
            @ produce_translate_matrix(part["joint_data_direction"], slide_distance)
        )

    posed = []
    for part in sorted(by_dfn.values(), key=lambda item: item["dfn"]):
        mesh = fit_mesh_to_bbx(part["mesh"], part["bbx"])
        vertices_h = np.concatenate([mesh.vertices, np.ones((len(mesh.vertices), 1))], axis=1)
        mesh.vertices = (part["transform"] @ vertices_h.T).T[:, :3]
        posed.append(mesh)
    return posed


def set_axes_equal(ax, meshes):
    bounds = np.array([mesh.bounds for mesh in meshes if len(mesh.vertices) > 0])
    xyz_min = bounds[:, 0, :].min(axis=0)
    xyz_max = bounds[:, 1, :].max(axis=0)
    span = np.maximum(xyz_max - xyz_min, 1e-3)
    margin = span * 0.08
    ax.set_xlim(xyz_min[0] - margin[0], xyz_max[0] + margin[0])
    ax.set_ylim(xyz_min[1] - margin[1], xyz_max[1] + margin[1])
    ax.set_zlim(xyz_min[2] - margin[2], xyz_max[2] + margin[2])
    ax.set_box_aspect(span)


def render_to_image(parts, ratio: float) -> Image.Image:
    meshes = apply_pose(parts, ratio)
    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)

    for idx, mesh in enumerate(meshes):
        vertices = mesh.vertices
        stride = max(1, len(vertices) // 18000)
        vertices = vertices[::stride]
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=0.45 if idx == 0 else 0.65,
            c=COLORS[idx % len(COLORS)],
            alpha=0.86,
            depthshade=False,
        )

    set_axes_equal(ax, meshes)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return Image.fromarray(image, mode="RGBA").convert("RGB")


def export_pose(parts, ratio: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_mesh"
    raw_dir.mkdir(exist_ok=True)

    meshes = apply_pose(parts, ratio)
    for idx, mesh in enumerate(meshes):
        mesh.export(raw_dir / f"{idx:02d}.obj")


def write_structure(parts, output_path: Path, source_meta, template_mode: str):
    serializable = []
    for part in parts:
        item = {
            key: part[key]
            for key in [
                "name",
                "dfn",
                "dfn_fa",
                "bbx",
                "joint_data_origin",
                "joint_data_direction",
                "limit",
                "source_mesh_path",
            ]
        }
        serializable.append(item)

    output_path.write_text(
        json.dumps(
            {
                "source_meta": source_meta,
                "template_mode": template_mode,
                "parts": serializable,
            },
            cls=HighPrecisionJsonEncoder,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Render a fixed 5-part articulated car template.")
    parser.add_argument("--shape_id", default="car_drivaer_117")
    parser.add_argument("--template_mode", choices=["canonical", "source-bbox"], default="canonical")
    parser.add_argument("--n_frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--wheel_turns", type=float, default=1.0)
    parser.add_argument("--loop_mode", choices=["cycle", "bounce"], default="bounce")
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--mesh_root", type=Path, default=env_path("CARACTGEN_MESH_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_mesh"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "fixed_template_eval",
    )
    args = parser.parse_args()

    source, source_parts = load_source_parts(args.shape_id, args.info_root, args.mesh_root)
    parts = build_parts(source_parts, args.template_mode)

    run_name = f"{time.strftime('%m-%d-%H%M%S')}_{args.shape_id}_{args.template_mode}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_structure(parts, output_dir / "structure.json", source["meta"], args.template_mode)
    with open(output_dir / "processed_nodes.pkl", "wb") as f:
        pickle.dump(parts, f)

    stills = {
        "pose_000.png": 0.0,
        "pose_050.png": args.wheel_turns * 0.5,
        "pose_100.png": args.wheel_turns,
    }
    for filename, ratio in stills.items():
        render_to_image(parts, ratio).save(output_dir / filename)
        export_pose(parts, ratio, output_dir / filename.replace(".png", ""))

    ratios = np.linspace(0.0, args.wheel_turns, args.n_frames)
    frames = [render_to_image(parts, float(ratio)) for ratio in ratios]
    if args.loop_mode == "bounce":
        frames = frames + list(reversed(frames))
    frames[0].save(
        output_dir / "motion.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
    )

    print(output_dir.as_posix())


if __name__ == "__main__":
    main()
