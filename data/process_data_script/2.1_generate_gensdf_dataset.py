
import os
import json
import shutil
import time
from pathlib import Path
from time import sleep
from rich import print
from rich.console import Console
from rich.table import Column, Table
from multiprocessing import Pool, cpu_count

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT.as_posix())
from utils.mylogging import console, Log

import trimesh
from tqdm import tqdm
import numpy as np

n_sample_point_each = None
uniform_sample_ratio = None
n_point_cloud = None

surface_point_sigma = 0.012
points_padding = 0.05
show_stats_table = False


def _round_to_multiple(value, multiple):
    value = int(round(value))
    multiple = int(multiple)
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(round(value / multiple)) * multiple)

def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def ply_to_obj(ply_file, obj_file):
    mesh = trimesh.load_mesh(
        open(ply_file.as_posix(), 'rb'),
        file_type='ply'
    )
    (min_bound, max_bound) = mesh.bounds
    center = (max_bound + min_bound) / 2
    mesh.vertices -= center

    scale = (max_bound - min_bound).max() / (2 - 0.01)
    mesh.vertices /= scale # fit into [-1, 1]

    mesh.export(obj_file.as_posix(), file_type='obj')

def obj_to_wtobj_by_pcu(obj_file, wt_obj_file, resolution=19_000):
    import point_cloud_utils as pcu
    import trimesh

    v, f = obj_to_wtobj_by_pcu_vf(obj_file, resolution=resolution)
    # Save via trimesh to avoid pcu PLY format issues
    wt_mesh = trimesh.Trimesh(vertices=v.astype('float32'), faces=f.astype('int32'))
    wt_mesh.export(wt_obj_file.as_posix())
    return "Done"

def obj_to_wtobj_by_pcu_vf(obj_file, resolution=19_000):
    import point_cloud_utils as pcu
    import trimesh

    # Use trimesh for mesh I/O, pcu only for watertight operation
    mesh = trimesh.load(obj_file.as_posix(), force='mesh')
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    # The resolution parameter controls the density of the output mesh. A
    # higher value preserves more thin detail but increases conversion time.
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    return vw, fw

# @see: https://www.fwilliams.info/point-cloud-utils/sections/mesh_sdf/
def wtobj_to_sdf_by_pcu(
    wt_obj_file,
    sdf_file,
    sample_method=[str, str],
    sample_config=None,
):
    '''
        Generate SDF from watertight obj file
        :param wt_obj_file: watertight obj file
        :param sdf_file: output sdf file
        :param sample_method: sample method for 'point near surface' and 'point cloud' respectively
                            choice: 'poisson_disk', 'uniform'
    '''
    import point_cloud_utils as pcu
    import trimesh

    # Use trimesh for mesh I/O to avoid pcu PLY format bugs
    mesh = trimesh.load(wt_obj_file.as_posix(), force='mesh')
    _v = np.asarray(mesh.vertices, dtype=np.float64)
    _f = np.asarray(mesh.faces, dtype=np.int32)

    sample_config = sample_config or {}
    n_point_total = int(sample_config.get("n_sample_point", n_sample_point_each))
    n_point_near_surface = int(n_point_total * (1 - uniform_sample_ratio))

    # Generate point near the surface
    Log.info('sampling point near mesh surface using %s', sample_method[0])

    if sample_method[0] == 'poisson_disk':
        fid, bc = pcu.sample_mesh_poisson_disk(_v, _f, num_samples=n_point_near_surface)
    elif sample_method[0] == 'random':
        fid, bc = pcu.sample_mesh_random(_v, _f, num_samples=n_point_near_surface)
    else:
        raise ValueError(f'Invalid method {sample_method[0]}')

    point_near_surface = pcu.interpolate_barycentric_coords(_f, fid, bc, _v)
    n_point_near_surface = point_near_surface.shape[0]
    point_near_surface += surface_point_sigma * np.random.randn(n_point_near_surface, 3)
    Log.info('point_near_surface: %s', point_near_surface.shape)

    # Generate point on the surface (point cloud)
    Log.info('sampling point on mesh surface using %s', sample_method[1])
    n_point_on_surface = int(sample_config.get("n_point_cloud", n_point_cloud))

    if sample_method[1] == 'poisson_disk':
        fid, bc = pcu.sample_mesh_poisson_disk(_v, _f, num_samples=n_point_on_surface)
    elif sample_method[1] == 'random':
        fid, bc = pcu.sample_mesh_random(_v, _f, num_samples=n_point_on_surface)
    else:
        raise ValueError(f'Invalid method {sample_method[1]}')

    point_on_surface = pcu.interpolate_barycentric_coords(_f, fid, bc, _v)
    ## In `interpolate_barycentric_coords()`, the number of points may not be exactly equal to `n_point_on_surface`
    n_point_on_surface = point_on_surface.shape[0]

    # Generate uniform point
    Log.info('sampling point in box')
    n_point_uniform = n_point_total - n_point_near_surface
    box_size = 2 + points_padding
    uniform_point = box_size * np.random.rand(n_point_uniform, 3) - (box_size / 2)

    # Combine surface and uniform point
    query_pts = np.concatenate([point_near_surface, uniform_point, point_on_surface], axis=0)

    Log.info('computing signed distance')
    sdf, fid, bc = pcu.signed_distance_to_mesh(query_pts, _v, _f)

    point_surface   = query_pts[:n_point_near_surface]
    sdf_surface     = sdf[:n_point_near_surface]

    point_uniform   = query_pts[n_point_near_surface:n_point_near_surface+n_point_uniform]
    sdf_uniform     = sdf[n_point_near_surface:n_point_near_surface+n_point_uniform]

    point_on        = query_pts[n_point_near_surface+n_point_uniform:]
    sdf_on          = sdf[n_point_near_surface+n_point_uniform:]

    assert (point_on.shape[0] == n_point_on_surface
        and sdf_on.shape[0] == n_point_on_surface
        and point_uniform.shape[0] == n_point_uniform
        and sdf_uniform.shape[0] == n_point_uniform
        and point_surface.shape[0] == n_point_near_surface
        and sdf_surface.shape[0] == n_point_near_surface), 'Error in point count'

    if show_stats_table:
        table = Table(show_header=True, header_style="bold magenta", title=wt_obj_file.stem)
        table.add_column("Item", justify="center")
        table.add_column("Shape", justify="center")
        table.add_column("Occ Rate", justify="center")
        table.add_column("Bounds", justify="center")
        table.add_column("Abs Sdf Range", justify="center")

        table.add_row("Point Uniform", str(point_uniform.shape), f"{(sdf_uniform < 0).astype(np.float32).mean():.4f}",
                      f"{point_uniform.min():.4f} ~ {point_uniform.max():.4f}", f"{np.abs(sdf_uniform).min():.4f} ~ {np.abs(sdf_uniform).max():.4f}")
        table.add_row("Point Surface", str(point_surface.shape), f"{(sdf_surface < 0).astype(np.float32).mean():.4f}",
                      f"{point_surface.min():.4f} ~ {point_surface.max():.4f}", f"{np.abs(sdf_surface).min():.4f} ~ {np.abs(sdf_surface).max():.4f}")
        table.add_row("Point On Mesh", str(point_on.shape), f"{(sdf_on < 0).astype(np.float32).mean():.4f}",
                      f"{point_on.min():.4f} ~ {point_on.max():.4f}", f"{np.abs(sdf_on).min():.4f} ~ {np.abs(sdf_on).max():.4f}")
        table.add_row("Total", str(query_pts.shape), f"{(sdf < 0).astype(np.float32).mean():.4f}",
                      f"{query_pts.min():.4f} ~ {query_pts.max():.4f}", f"{np.abs(sdf).min():.4f} ~ {np.abs(sdf).max():.4f}")

        console.print(table)

    np.savez(sdf_file.as_posix(),
             point_uniform=point_uniform, sdf_uniform=sdf_uniform,
             point_surface=point_surface, sdf_surface=sdf_surface,
             point_on=point_on, sdf_on=sdf_on,
             sampling_meta=json.dumps(sample_config))

    return "Done"


def convert_mesh(
    ply_file,
    clear_temp,
    wt_method,
    sdf_method,
    output_dir,
    temp_root,
    sample_method=['random', 'random'],
    sample_config=None,
):
    start_time = time.time()
    stem = ply_file.stem
    temp_dir = Path(temp_root) / stem
    result_dir = Path(output_dir)

    create_folder(temp_dir, result_dir)

    obj_file = temp_dir / (stem + ".obj")
    wt_obj_file = temp_dir / (stem + ".wt.ply")
    sdf_target_file = result_dir / f'{stem}.sdf'

    if (result_dir / f'{stem}.sdf.npz').exists():
        Log.info('Already exists: %s', sdf_target_file)
        return "Done"

    try:
        Log.info('(1) Converting to (obj) %s', obj_file)
        ply_to_obj(ply_file, obj_file)

        Log.info('(2) Converting to (wt) %s', wt_obj_file)
        if wt_method == 'pcu':
            Log.info('(2) Using pcu to watertight obj')
            watertight_resolution = (sample_config or {}).get('watertight_resolution', 19_000)
            assert obj_to_wtobj_by_pcu(obj_file, wt_obj_file, resolution=watertight_resolution) == 'Done'
        else:
            raise ValueError(f'Invalid method wt_method {wt_method}')

        Log.info('(3) Converting to (sdf) %s', sdf_target_file)
        if sdf_method == 'pcu':
            Log.info('(3) Using pcu to generate sdf')
            if wtobj_to_sdf_by_pcu(wt_obj_file, sdf_target_file, sample_method, sample_config=sample_config) != 'Done':
                raise RuntimeError('Error in sdf generation')
        else:
            raise ValueError(f'Invalid method sdf_method {sdf_method}')

        Log.info('finished in %.2f s', time.time() - start_time)
        if clear_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return "Done"
    except Exception as exc:
        Log.error('Failed converting %s: %s', ply_file, exc)
        return f"Error: {exc}"


def measure_source_mesh_stats(ply_files):
    stats = {}
    for ply_file in tqdm(ply_files, desc='Measuring source meshes'):
        mesh = trimesh.load_mesh(open(ply_file.as_posix(), 'rb'), file_type='ply', process=False)
        stats[ply_file.stem] = {
            'path': ply_file.as_posix(),
            'area': float(mesh.area),
            'faces': int(len(mesh.faces)),
            'vertices': int(len(mesh.vertices)),
            'extents': [float(v) for v in mesh.extents],
            'is_watertight': bool(mesh.is_watertight),
        }
    return stats


def load_or_measure_source_mesh_stats(ply_files, stats_json_path=None):
    stats_json_path = Path(stats_json_path) if stats_json_path else None
    if stats_json_path and stats_json_path.exists():
        stats = json.loads(stats_json_path.read_text())
        missing = [p for p in ply_files if p.stem not in stats]
        if not missing:
            return stats
        Log.warning('Stats file is missing %s meshes; recomputing stats.', len(missing))

    stats = measure_source_mesh_stats(ply_files)
    if stats_json_path:
        stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        stats_json_path.write_text(json.dumps(stats, indent=2))
    return stats


def build_adaptive_sample_plan(ply_files, stats, args):
    areas = np.array([max(float(stats[p.stem]['area']), 1e-8) for p in ply_files], dtype=np.float64)
    faces = np.array([max(float(stats[p.stem]['faces']), 1.0) for p in ply_files], dtype=np.float64)
    ref_area = float(np.median(areas))
    ref_faces = float(np.median(faces))

    plan = {}
    for ply_file in ply_files:
        stat = stats[ply_file.stem]
        area_score = (max(float(stat['area']), 1e-8) / ref_area) ** float(args.adaptive_area_power)
        face_score = (max(float(stat['faces']), 1.0) / ref_faces) ** float(args.adaptive_face_power)
        denom = float(args.adaptive_area_weight) + float(args.adaptive_face_weight)
        if denom <= 0:
            multiplier = 1.0
        else:
            multiplier = (
                float(args.adaptive_area_weight) * area_score
                + float(args.adaptive_face_weight) * face_score
            ) / denom
        multiplier = float(np.clip(multiplier, args.adaptive_min_multiplier, args.adaptive_max_multiplier))

        n_sample = _round_to_multiple(args.n_sample_point * multiplier, args.adaptive_round_to)
        n_pc = _round_to_multiple(args.n_point_cloud * multiplier, args.adaptive_round_to)

        wt_multiplier = multiplier if args.adaptive_watertight_resolution else 1.0
        wt_resolution = _round_to_multiple(args.watertight_resolution * wt_multiplier, args.adaptive_round_to)
        wt_resolution = int(np.clip(wt_resolution, args.watertight_resolution, args.max_watertight_resolution))

        plan[ply_file.stem] = {
            'adaptive_sampling': True,
            'source_area': float(stat['area']),
            'source_faces': int(stat['faces']),
            'source_vertices': int(stat['vertices']),
            'adaptive_reference_area': ref_area,
            'adaptive_reference_faces': ref_faces,
            'adaptive_multiplier': multiplier,
            'n_sample_point': int(n_sample),
            'n_point_cloud': int(n_pc),
            'watertight_resolution': int(wt_resolution),
        }
    return plan


def build_fixed_sample_plan(ply_files, args):
    return {
        ply_file.stem: {
            'adaptive_sampling': False,
            'n_sample_point': int(args.n_sample_point),
            'n_point_cloud': int(args.n_point_cloud),
            'watertight_resolution': int(args.watertight_resolution),
        }
        for ply_file in ply_files
    }


def print_sample_plan_summary(plan):
    rows = list(plan.items())
    samples = np.array([v['n_sample_point'] for _, v in rows], dtype=np.float64)
    pcs = np.array([v['n_point_cloud'] for _, v in rows], dtype=np.float64)
    mult = np.array([v.get('adaptive_multiplier', 1.0) for _, v in rows], dtype=np.float64)
    Log.info(
        'Sample plan: files=%s n_sample[min/median/max]=%s/%s/%s pc[min/median/max]=%s/%s/%s multiplier[min/median/max]=%.3f/%.3f/%.3f',
        len(rows),
        int(samples.min()), int(np.median(samples)), int(samples.max()),
        int(pcs.min()), int(np.median(pcs)), int(pcs.max()),
        float(mult.min()), float(np.median(mult)), float(mult.max()),
    )
    for stem, cfg in sorted(rows, key=lambda kv: kv[1].get('adaptive_multiplier', 1.0), reverse=True)[:10]:
        Log.info(
            'Top adaptive mesh %s: sample=%s pc=%s multiplier=%.3f area=%.4f faces=%s',
            stem,
            cfg['n_sample_point'],
            cfg['n_point_cloud'],
            cfg.get('adaptive_multiplier', 1.0),
            cfg.get('source_area', -1.0),
            cfg.get('source_faces', -1),
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate SDF from mesh files.')
    default_data_root = Path(os.environ.get('CARACTGEN_DATA_ROOT', '../datasets'))
    parser.add_argument('--input_mesh_dir', type=Path, default=default_data_root / '1_preprocessed_mesh', help='Input mesh directory.')
    parser.add_argument('--output_dir', type=Path, default=default_data_root / '2_gensdf_dataset', help='Output directory for SDF npz files.')
    parser.add_argument('--meta_json_path', type=Path, default=default_data_root / 'meta.json', help='Dataset meta.json path.')
    parser.add_argument('--clear_temp_file', action='store_true', help='Clear temp files after each success.')
    parser.add_argument('--reset_output', action='store_true', help='Delete the whole output directory before generation.')
    parser.add_argument('--log_per_mesh_stats', action='store_true', help='Print per-mesh point statistics tables.')
    parser.add_argument('--limit', type=int, default=-1, help='Optional limit for the number of pending meshes to process.')

    parser.add_argument('--n_sample_point', type=int, default=400000, help='Number of points for each mesh')
    parser.add_argument('--uniform_sample_ratio', type=float, default=0.5, help='Uniform sample ratio')
    parser.add_argument('--n_point_cloud', type=int, default=100000, help='Number of point cloud')
    parser.add_argument('--watertight_resolution', type=int, default=19000, help='Resolution for pcu.make_mesh_watertight.')
    parser.add_argument('--n_process', type=int, default=10, help='Number of process')
    parser.add_argument('--near_surface_sammple_method', type=str, default='random', help='Sample method for near surface', choices=['poisson_disk', 'random'])
    parser.add_argument('--on_surface_sample_method', type=str,  default='poisson_disk', help='Sample method for on surface', choices=['poisson_disk', 'random'])
    parser.add_argument('--adaptive_sampling', action='store_true', help='Scale samples per mesh by source mesh area and face count.')
    parser.add_argument('--adaptive_stats_json', type=Path, default=None, help='Optional cache path for source mesh statistics.')
    parser.add_argument('--adaptive_plan_json', type=Path, default=None, help='Optional path to save the resolved sample plan.')
    parser.add_argument('--adaptive_area_weight', type=float, default=0.75, help='Weight of source area in adaptive sampling.')
    parser.add_argument('--adaptive_face_weight', type=float, default=0.25, help='Weight of source face count in adaptive sampling.')
    parser.add_argument('--adaptive_area_power', type=float, default=0.5, help='Exponent applied to source area ratio.')
    parser.add_argument('--adaptive_face_power', type=float, default=0.5, help='Exponent applied to source face-count ratio.')
    parser.add_argument('--adaptive_min_multiplier', type=float, default=1.0, help='Lower bound for adaptive sample multiplier.')
    parser.add_argument('--adaptive_max_multiplier', type=float, default=5.0, help='Upper bound for adaptive sample multiplier.')
    parser.add_argument('--adaptive_round_to', type=int, default=1024, help='Round adaptive counts to this multiple.')
    parser.add_argument('--adaptive_watertight_resolution', action='store_true', help='Also scale watertight conversion resolution adaptively.')
    parser.add_argument('--max_watertight_resolution', type=int, default=30000, help='Upper bound for adaptive watertight resolution.')
    parser.add_argument('--dry_run', action='store_true', help='Only resolve and print the sample plan; do not generate SDF files.')

    args = parser.parse_args()

    n_sample_point_each = args.n_sample_point
    uniform_sample_ratio = args.uniform_sample_ratio
    n_point_cloud = args.n_point_cloud
    sample_method = [args.near_surface_sammple_method, args.on_surface_sample_method]
    show_stats_table = args.log_per_mesh_stats

    input_mesh_dir = args.input_mesh_dir
    output_dir = args.output_dir
    temp_root = output_dir / 'temp'

    if args.reset_output:
        shutil.rmtree(output_dir, ignore_errors=True)
    create_folder(output_dir, temp_root)

    all_ply_files = sorted(input_mesh_dir.glob('*.ply'))
    if not all_ply_files:
        raise FileNotFoundError(f'No .ply files found in {input_mesh_dir}')

    if args.adaptive_sampling:
        stats = load_or_measure_source_mesh_stats(all_ply_files, args.adaptive_stats_json)
        sample_plan = build_adaptive_sample_plan(all_ply_files, stats, args)
    else:
        sample_plan = build_fixed_sample_plan(all_ply_files, args)
    print_sample_plan_summary(sample_plan)
    if args.adaptive_plan_json:
        args.adaptive_plan_json.parent.mkdir(parents=True, exist_ok=True)
        args.adaptive_plan_json.write_text(json.dumps(sample_plan, indent=2))
    if args.dry_run:
        Log.info('Dry run requested; exiting before conversion.')
        raise SystemExit(0)

    existing_outputs = [
        ply_file for ply_file in all_ply_files
        if (output_dir / f'{ply_file.stem}.sdf.npz').exists()
    ]
    pending_ply_files = [
        ply_file for ply_file in all_ply_files
        if not (output_dir / f'{ply_file.stem}.sdf.npz').exists()
    ]
    if args.limit > 0:
        pending_ply_files = pending_ply_files[:args.limit]

    print("all_ply_files: ", len(all_ply_files))
    print("existing_outputs: ", len(existing_outputs))
    print("pending_ply_files: ", len(pending_ply_files))

    failed = []
    done = []
    if pending_ply_files:
        with Pool(args.n_process) as p:
            result = [
                (
                    p.apply_async(
                        convert_mesh,
                        (
                            ply_file,
                            args.clear_temp_file,
                            'pcu',
                            'pcu',
                            output_dir,
                            temp_root,
                            sample_method,
                            sample_plan[ply_file.stem],
                        ),
                    ),
                    ply_file,
                )
                for ply_file in pending_ply_files
            ]
            bar = tqdm(total=len(result), desc='Converting meshes')
            while result:
                for r, ply_file in result[:]:
                    if r.ready():
                        bar.update(1)
                        try:
                            status = r.get()
                        except Exception as exc:
                            status = f"Error: {exc}"
                        if status != 'Done':
                            Log.error('Error in converting mesh %s: %s', ply_file, status)
                            failed.append({'file': str(ply_file), 'error': status})
                        else:
                            Log.info('Done: %s', ply_file)
                            done.append(ply_file)
                        result.remove((r, ply_file))
                sleep(0.1)

    meta = {}
    if args.meta_json_path.exists():
        meta = json.loads(args.meta_json_path.read_text())

    final_output_count = len(list(output_dir.glob('*.sdf.npz')))

    meta['2_generate_gensdf_dataset'] = {
        'create_date_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_mesh_dir': str(input_mesh_dir),
        'output_dir': str(output_dir),
        'reset_output': args.reset_output,
        'n_sample_point': n_sample_point_each,
        'uniform_sample_ratio': uniform_sample_ratio,
        'n_point_cloud': n_point_cloud,
        'watertight_resolution': args.watertight_resolution,
        'adaptive_sampling': args.adaptive_sampling,
        'adaptive_area_weight': args.adaptive_area_weight,
        'adaptive_face_weight': args.adaptive_face_weight,
        'adaptive_area_power': args.adaptive_area_power,
        'adaptive_face_power': args.adaptive_face_power,
        'adaptive_min_multiplier': args.adaptive_min_multiplier,
        'adaptive_max_multiplier': args.adaptive_max_multiplier,
        'adaptive_round_to': args.adaptive_round_to,
        'adaptive_watertight_resolution': args.adaptive_watertight_resolution,
        'max_watertight_resolution': args.max_watertight_resolution,
        'adaptive_plan_json': str(args.adaptive_plan_json) if args.adaptive_plan_json else None,
        'adaptive_stats_json': str(args.adaptive_stats_json) if args.adaptive_stats_json else None,
        'near_surface_sammple_method': sample_method[0],
        'on_surface_sample_method': sample_method[1],
        'all_input_files': [str(f) for f in all_ply_files],
        'all_input_files_count': len(all_ply_files),
        'existing_outputs_before_count': len(existing_outputs),
        'pending_count': len(pending_ply_files),
        'done': [str(f) for f in done],
        'done_count': len(done),
        'failed': failed,
        'failed_count': len(failed),
        'final_output_count': final_output_count,
        'done_ratio': final_output_count / len(all_ply_files)
    }

    with open(args.meta_json_path, 'w') as f:
        f.write(json.dumps(meta, indent=2))

    Log.info('all_ply_files = %s.', len(all_ply_files))
    Log.info('Current Done = %s.', final_output_count)
    Log.critical('Failed: %s', failed)
