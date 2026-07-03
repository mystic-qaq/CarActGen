"""
Multi-GPU parallel latent extraction: split 2420 parts across 8 GPUs.
Each GPU loads its own SDF model and processes its share independently.
"""
import json, os, sys, math, shutil
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '../..')
from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.dataloader import GenSDFDataset
from torch.utils.data import DataLoader, Subset

DEFAULT_DATA_ROOT = Path(os.environ.get('CARACTGEN_DATA_ROOT', '../datasets'))
CKPT = os.environ.get('CARACTGEN_ORIGINAL_VAE_CKPT')
OUTPUT_DIR = Path(os.environ.get('CARACTGEN_TEXT_LATENT_ROOT', DEFAULT_DATA_ROOT / '2.1_text_n_latentcode'))


def process_gpu_chunk(gpu_id: int, file_indices: list[int], all_files: list):
    """Process a subset of files on a specific GPU."""
    if not CKPT:
        raise ValueError('Set CARACTGEN_ORIGINAL_VAE_CKPT or pass a checkpoint through the environment.')
    device = torch.device(f'cuda:{gpu_id}')
    print(f'[GPU {gpu_id}] Loading model...', flush=True)
    model = SDFAutoEncoder.load_from_checkpoint(CKPT, map_location=device)
    model = model.to(device)
    model.eval()

    results = {}
    batch_size = 24  # smaller batch since single GPU

    for idx in tqdm(file_indices, desc=f'GPU {gpu_id}', position=gpu_id):
        npz_path = all_files[idx]
        try:
            data = np.load(npz_path)
            pc = torch.from_numpy(data['point_on'][:4096]).float().unsqueeze(0).to(device)

            with torch.no_grad():
                plane_features = model.encoder.get_plane_features(pc)
                original_features = torch.cat(plane_features, dim=1)
                out = model.vae_model(original_features)
                z = out[2]  # mu

            latent = z.squeeze(0).cpu().numpy()
            stem = Path(npz_path).stem.replace('.sdf', '')
            results[stem] = latent
        except Exception as e:
            print(f'[GPU {gpu_id}] Error on {npz_path}: {e}', flush=True)

    # Save results for this GPU chunk
    out_path = OUTPUT_DIR / f'latents_gpu{gpu_id}.npz'
    np.savez(out_path, **results)
    print(f'[GPU {gpu_id}] Done: {len(results)} latents saved to {out_path}', flush=True)
    return len(results)


def main():
    # Prepare output
    shutil.rmtree(str(OUTPUT_DIR), ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all NPZ files
    dataset = GenSDFDataset(
        dataset_dir=DEFAULT_DATA_ROOT,
        train=None, samples_per_mesh=16000, pc_size=4096, uniform_sample_ratio=0.3
    )
    all_files = [str(f) for f in dataset.dataset_dir]
    print(f'Total files: {len(all_files)}')

    # Split across GPUs
    n_gpus = 8
    chunks = np.array_split(np.arange(len(all_files)), n_gpus)
    chunks = [c.tolist() for c in chunks]
    for i, c in enumerate(chunks):
        print(f'GPU {i}: {len(c)} files (first={Path(all_files[c[0]]).name}, last={Path(all_files[c[-1]]).name})')

    # Launch processes
    with mp.Pool(n_gpus) as pool:
        results = [pool.apply_async(process_gpu_chunk, (i, chunk, all_files))
                   for i, chunk in enumerate(chunks)]
        total = sum(r.get() for r in results)

    # Merge all chunks
    print(f'\nMerging {total} latents from {n_gpus} chunks...')
    all_latents = {}
    for gpu_id in range(n_gpus):
        chunk_path = OUTPUT_DIR / f'latents_gpu{gpu_id}.npz'
        if chunk_path.exists():
            data = dict(np.load(chunk_path, allow_pickle=True))
            all_latents.update(data)
            chunk_path.unlink()  # delete chunk

    print(f'Total latents: {len(all_latents)}')

    # Generate text labels (offline dummy for now)
    mesh_info_path = Path(os.environ.get('CARACTGEN_INFO_ROOT', DEFAULT_DATA_ROOT / '1_preprocessed_info'))
    text_labels = set()
    for shape_json_path in mesh_info_path.glob('*.json'):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = shape_json['meta']['catecory']
        for part_info in shape_json['part']:
            text_labels.add(f"{shape_name}, {part_info['name']}")

    # Dummy text encoding (real T5 can be applied later)
    rng = np.random.RandomState(42)
    text_to_emb = {t: rng.randn(1024).astype(np.float32) for t in text_labels}

    # Save individual NPZ files
    failed, success = [], []
    for shape_json_path in mesh_info_path.glob('*.json'):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = shape_json['meta']['catecory']
        for part_info in shape_json['part']:
            mesh_name = Path(part_info['mesh']).stem
            if mesh_name not in all_latents:
                failed.append(mesh_name)
                continue
            np.savez(OUTPUT_DIR / f'{mesh_name}.npz',
                     latent_code=all_latents[mesh_name],
                     text=text_to_emb[f"{shape_name}, {part_info['name']}"],
                     text_label=np.array(f"{shape_name}, {part_info['name']}", dtype=str))
            success.append(mesh_name)

    with open(OUTPUT_DIR / 'meta.json', 'w') as f:
        json.dump({'ckpt': CKPT, 'success_count': len(success), 'failed': failed}, f, indent=2)

    print(f'Success: {len(success)}, Failed: {len(failed)}')
    print('Done!')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
