"""
Extract car dataset (body+4 wheels) into ArtFormer preprocessed format.
Directly generates 1_preprocessed_info + copies meshes to 1_preprocessed_mesh.
"""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import random
# Inline to avoid importing utils (which requires pyvista)
class HighPrecisionJsonEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, float):
            return format(obj, '.40f')
        return json.JSONEncoder.encode(self, obj)


def build_dfn_map(parts: list[dict]):
    """Normalize car part ids to ArtFormer's virtual-root convention."""
    root_candidates = [
        p for p in parts
        if p.get('dfn_fa') in (-1, None) or p.get('joint') is None or p.get('name') == 'body_shell'
    ]
    if not root_candidates:
        raise ValueError("No root/body part found")

    root = sorted(root_candidates, key=lambda p: (p.get('name') != 'body_shell', p['dfn']))[0]
    old_root_dfn = root['dfn']

    dfn_map = {old_root_dfn: 1}
    children = [p for p in parts if p['dfn'] != old_root_dfn]
    children.sort(key=lambda p: (p.get('dfn', 0), p.get('name', '')))
    for idx, part in enumerate(children, start=2):
        dfn_map[part['dfn']] = idx

    return dfn_map, old_root_dfn


def normalize_revolute_limit(limit):
    """Return [slide_min, slide_max, rot_min, rot_max] with rotation in radians."""
    rot_min, rot_max = float(limit[0]), float(limit[1])
    if max(abs(rot_min), abs(rot_max)) > 2 * math.pi + 1e-4:
        rot_min, rot_max = math.radians(rot_min), math.radians(rot_max)
    return [0., 0., rot_min, rot_max]


def process(shape_path: Path, output_info_path: Path, output_mesh_path: Path):
    """Convert one car's meta.json → preprocessed info format."""
    start_time = time.time()
    raw_meta = json.loads((shape_path / 'meta.json').read_text())
    meta = raw_meta['meta']
    catecory_name = meta['catecory']
    key_name = f"{catecory_name}_{meta['shape_id']}"

    processed_part = []

    dfn_map, old_root_dfn = build_dfn_map(raw_meta['part'])

    for part in raw_meta['part']:
        mesh_src = shape_path / part['mesh']
        mesh_dst_name = f"{key_name}_{part['dfn']}.ply"
        mesh_dst = output_mesh_path / mesh_dst_name

        # Convert OBJ → PLY using trimesh
        if not mesh_dst.exists():
            import trimesh
            m = trimesh.load(str(mesh_src))
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(m.dump())
            m.export(str(mesh_dst))

        joint = part.get('joint')
        if joint is None:
            joint_data_origin = [0, 0, 0]
            joint_data_direction = [0, 0, 0]
            limit = [0., 0., 0., 0.]
        else:
            joint_data_origin = joint['origin']
            joint_data_direction = joint['direction']
            lim = joint.get('limit', [0, 0])
            limit = normalize_revolute_limit(lim)

        new_dfn = dfn_map[part['dfn']]
        if part['dfn'] == old_root_dfn:
            new_dfn_fa = 0
        else:
            new_dfn_fa = dfn_map.get(part['dfn_fa'], 1)

        processed_part.append({
            'name': part['name'],
            'dfn': new_dfn,
            'dfn_fa': new_dfn_fa,
            'mesh': mesh_dst_name,
            'bbx': part['bbx'],
            'joint_data_origin': joint_data_origin,
            'joint_data_direction': joint_data_direction,
            'limit': limit,
        })

    output_info_path_file = output_info_path / (key_name + ".json")
    output_info_path_file.write_text(json.dumps({
        'meta': meta,
        'part': processed_part
    }, cls=HighPrecisionJsonEncoder, indent=2))

    end_time = time.time()
    return f'[Done] {key_name} ({len(processed_part)} parts) time: {end_time - start_time:.2f}s', key_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract a 5-part car dataset into ArtFormer preprocessed format.")
    parser.add_argument(
        "--raw_dataset_root",
        type=Path,
        default=Path(os.environ.get("CARACTGEN_RAW_CAR_ROOT", "../datasets/0_raw_dataset/car")),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(os.environ.get("CARACTGEN_DATA_ROOT", "../datasets")),
    )
    parser.add_argument("--n_process", type=int, default=32)
    parser.add_argument("--reset_output", action="store_true")
    args = parser.parse_args()

    raw_dataset_paths = sorted(glob((args.raw_dataset_root / "*").as_posix()))
    output_info_path = args.output_root / "1_preprocessed_info"
    output_mesh_path = args.output_root / "1_preprocessed_mesh"
    train_split_ratio = 1

    if args.reset_output:
        shutil.rmtree(str(output_info_path), ignore_errors=True)
        shutil.rmtree(str(output_mesh_path), ignore_errors=True)
    output_info_path.mkdir(exist_ok=True, parents=True)
    output_mesh_path.mkdir(exist_ok=True, parents=True)

    failed_shape_path = {}
    success_shape_path = {}

    with Pool(args.n_process) as p:
        results = [
            p.apply_async(process, (Path(sp), output_info_path, output_mesh_path))
            for sp in tqdm(raw_dataset_paths, desc="Queuing cars")
        ]

        bar = tqdm(total=len(raw_dataset_paths), desc='Processing cars')
        while results:
            for r in list(results):
                if not r.ready():
                    continue
                bar.update(1)
                status = r.get()
                if 'Error' in status[0]:
                    failed_shape_path[status[1]] = status[0]
                elif 'Done' in status[0]:
                    success_shape_path[status[1]] = status[0]
                results.remove(r)
            time.sleep(0.05)

    success_shape_key_name = list(success_shape_path.keys())
    random.shuffle(success_shape_key_name)

    train_split = success_shape_key_name[:int(len(success_shape_key_name) * train_split_ratio)]
    test_split = success_shape_key_name[int(len(success_shape_key_name) * train_split_ratio):]

    print(f'Failed: {len(failed_shape_path)}')
    print(f'Success: {len(success_shape_path)}')
    for f in failed_shape_path.values():
        print(f'  {f}')

    meta_info = args.output_root / 'meta.json'
    with open(meta_info, 'w') as f:
        json.dump({"1_extract_from_raw_dataset": {
            'created_date_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'train_split': train_split,
            'test_split': test_split,
            'train_split_count': len(train_split),
            'test_split_count': len(test_split),
            'selected_categories': ['car'],
            'success_shape_count': len(success_shape_path),
            'failed_shape_count': len(failed_shape_path),
            'category_count': {'car': len(success_shape_path)}
        }}, f, indent=2)
