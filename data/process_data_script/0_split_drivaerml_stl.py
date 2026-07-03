"""
Split DrivAerML ASCII STL (49 solids, ~135MB each) into PartNet-Mobility format.
Each car → 5 parts: body (all non-wheel solids) + 4 wheels (Tire+Rim+BrakeDisc+TirePlinth).
Output: <dataset_root>/0_raw_dataset/car/
"""
import argparse
import json
import os
import shutil
import sys
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import numpy as np
import trimesh
from tqdm import tqdm

# === CONFIG ===
STL_DATASET = Path(os.environ.get("DRIVAERML_ROOT", "../datasets/drivaerml"))
OUTPUT_DIR = Path(os.environ.get("CARACTGEN_DATA_ROOT", "../datasets")) / "0_raw_dataset/car"

# Solids that contain rotating wheel geometry.
# Each solid typically has 2 components (left/right) split by Y sign.
# TirePlinth has 6 small components (3 per side) = ground-contact patches.
WHEEL_SOLID_NAMES = {
    "Tiresfront":       "front",
    "Tiresrear":        "rear",
    "Rimsfront":        "front",
    "Rimsrear":         "rear",
    "BrakeDiscfront":   "front",
    "BrakeDiscrear":    "rear",
    "TirePlinthfront":  "front",
    "TirePlinthrear":   "rear",
}

SHORT_KEYS = ["front_left", "front_right", "rear_left", "rear_right"]
WHEEL_FULL_NAMES = {
    "front_left":  "wheel_front_left",
    "front_right": "wheel_front_right",
    "rear_left":   "wheel_rear_left",
    "rear_right":  "wheel_rear_right",
}

ROTATION_AXIS = [0.0, 1.0, 0.0]

# Minimum faces for a component to be considered valid
MIN_WHEEL_COMP_FACES = 20   # TirePlinth patches are ~90 faces
MIN_BODY_COMP_FACES = 50


def _valid_mesh(m):
    try:
        if m.faces.shape[0] == 0 or m.vertices.shape[0] == 0:
            return False
        if not np.all(np.isfinite(m.vertices)):
            return False
        return True
    except Exception:
        return False


def _safe_centroid(m):
    try:
        c = m.centroid
        if c is not None and np.all(np.isfinite(c)):
            return c
    except Exception:
        pass
    return np.zeros(3)


def parse_stl_solids(stl_path):
    scene = trimesh.load(str(stl_path))
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError(f"Expected Scene, got {type(scene)}")
    return dict(scene.geometry)


def split_wheel_by_position(components, position):
    """Classify wheel components by left(Y<0)/right(Y>0) + front/rear."""
    result = defaultdict(list)
    for comp in components:
        cy = _safe_centroid(comp)[1]
        if position == "front":
            side = "front_left" if cy < 0 else "front_right"
        else:
            side = "rear_left" if cy < 0 else "rear_right"
        result[side].append(comp)
    return result


def combine_meshes(meshes):
    if len(meshes) == 0:
        raise ValueError("No meshes to combine")
    if len(meshes) == 1:
        return meshes[0]
    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices()
    return combined


def process_car(car_id):
    stl_path = STL_DATASET / f"run_{car_id}" / f"drivaer_{car_id}.stl"
    if not stl_path.exists():
        return f"[Skip] run_{car_id}: STL not found"

    car_name = f"drivaer_{car_id}"
    car_output_dir = OUTPUT_DIR / car_name
    car_output_dir.mkdir(parents=True, exist_ok=True)

    solids = parse_stl_solids(stl_path)
    wheel_components = defaultdict(list)
    body_meshes = []

    for solid_name, mesh in solids.items():
        if solid_name in WHEEL_SOLID_NAMES:
            position = WHEEL_SOLID_NAMES[solid_name]
            components = mesh.split(only_watertight=False)
            valid = [c for c in components if _valid_mesh(c) and c.faces.shape[0] >= MIN_WHEEL_COMP_FACES]
            if not valid:
                body_meshes.append(mesh)
                continue
            classified = split_wheel_by_position(valid, position)
            for corner, comps in classified.items():
                wheel_components[corner].extend(comps)
        else:
            components = mesh.split(only_watertight=False)
            large = [c for c in components if _valid_mesh(c) and c.faces.shape[0] > MIN_BODY_COMP_FACES]
            if large:
                body_meshes.extend(large)
            else:
                body_meshes.append(mesh)

    # --- Body ---
    if not body_meshes:
        return f"[Error] {car_name}: no body meshes"
    body_mesh = combine_meshes(body_meshes)
    body_dst = car_output_dir / "body_shell.obj"
    body_mesh.export(str(body_dst))
    body_bbox = [body_mesh.bounds[0].tolist(), body_mesh.bounds[1].tolist()]

    parts = [{
        "name": "body_shell", "dfn": 1, "dfn_fa": 0,
        "mesh": "body_shell.obj", "bbx": body_bbox, "joint": None
    }]

    # --- 4 Wheels ---
    # ArtFormer convention: root body is dfn=1, wheels are dfn=2..5, parent=1
    import math
    for idx, short_key in enumerate(SHORT_KEYS):
        full_name = WHEEL_FULL_NAMES[short_key]
        if short_key not in wheel_components or len(wheel_components[short_key]) == 0:
            return f"[Error] {car_name}: {full_name} has no components"

        wheel_mesh = combine_meshes(wheel_components[short_key])
        wheel_dst = car_output_dir / f"{full_name}.obj"
        wheel_mesh.export(str(wheel_dst))
        wheel_bbox = [wheel_mesh.bounds[0].tolist(), wheel_mesh.bounds[1].tolist()]
        rotation_origin = _safe_centroid(wheel_mesh).tolist()

        parts.append({
            "name": full_name, "dfn": idx + 2, "dfn_fa": 1,
            "mesh": f"{full_name}.obj", "bbx": wheel_bbox,
            "joint": {
                "type": "revolute",
                "origin": rotation_origin,
                "direction": ROTATION_AXIS,
                "limit": [0.0, 2 * math.pi]
            }
        })

    meta = {"meta": {"catecory": "car", "shape_id": car_name}, "part": parts}
    (car_output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    bv, bf = body_mesh.vertices.shape[0], body_mesh.faces.shape[0]
    return f"[OK] {car_name}: body={bv}v/{bf}f"


def main():
    global STL_DATASET, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Split DrivAerML STL files into 5-part articulated car assets.")
    parser.add_argument("--stl_dataset", type=Path, default=STL_DATASET)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n_workers", type=int, default=min(64, mp.cpu_count()))
    parser.add_argument("--reset_output", action="store_true")
    args = parser.parse_args()

    STL_DATASET = args.stl_dataset
    OUTPUT_DIR = args.output_dir

    if args.reset_output and OUTPUT_DIR.exists():
        shutil.rmtree(str(OUTPUT_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    car_ids = []
    for run_dir in sorted(STL_DATASET.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            try:
                car_ids.append(int(run_dir.name.split("_")[1]))
            except ValueError:
                continue
    car_ids.sort()
    print(f"Found {len(car_ids)} cars")

    n_workers = args.n_workers
    print(f"Using {n_workers} workers (CPU cores: {mp.cpu_count()})")

    failed = []
    with mp.Pool(n_workers) as pool:
        results = [pool.apply_async(process_car, (cid,)) for cid in car_ids]
        for r in tqdm(results, desc="Splitting STL"):
            status = r.get()
            tqdm.write(status)
            if "[Error]" in status or "[Skip]" in status:
                failed.append(status)

    n_ok = len(car_ids) - len(failed)
    print(f"\nDone: {n_ok}/{len(car_ids)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    # Write dataset info
    info_path = OUTPUT_DIR.parent / "0_raw_dataset_info.json"
    info_path.write_text(json.dumps({
        "category": "car", "count": n_ok,
        "raw_dir": str(OUTPUT_DIR), "source": str(STL_DATASET)
    }, indent=2))


if __name__ == "__main__":
    main()
