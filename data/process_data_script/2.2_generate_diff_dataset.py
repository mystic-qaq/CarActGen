import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

sys.path.append('../..')
from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.dataloader import GenSDFDataset
from utils import camel_to_snake, to_cuda
from utils.mylogging import Log


def determine_latentcode_encoder(best_ckpt_path: Path, device: torch.device):
    Log.info('Using best ckpt: %s', best_ckpt_path)
    gensdf = SDFAutoEncoder.load_from_checkpoint(best_ckpt_path, map_location=device)
    return gensdf


def evaluate_latent_codes(
    gensdf,
    dataset_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    samples_per_mesh: int,
    pc_size: int,
    uniform_sample_ratio: float,
):
    dataloader = DataLoader(
        GenSDFDataset(
            dataset_dir=dataset_dir,
            train=None,
            samples_per_mesh=samples_per_mesh,
            pc_size=pc_size,
            uniform_sample_ratio=uniform_sample_ratio,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print("Length =", len(dataloader.dataset))
    gensdf.eval()
    gensdf = gensdf.to(device)

    path_to_latent = {}
    for _, batched_data in tqdm(
        enumerate(dataloader),
        desc='Evaluating Latent Code',
        total=len(dataloader),
    ):
        x = to_cuda(batched_data)
        pc = x['point_cloud']

        with torch.no_grad():
            plane_features = gensdf.encoder.get_plane_features(pc)
            original_features = torch.cat(plane_features, dim=1)
            out = gensdf.vae_model(original_features)  # [decode(z), input, mu, log_var, z]

        latent_tensor = out[2]  # use posterior mean as a deterministic diffusion target

        for batch_idx in range(latent_tensor.shape[0]):
            latent = latent_tensor[batch_idx, ...]
            latent_numpy = latent.detach().cpu().numpy()
            path = x['filename'][batch_idx]
            path = Path(path).stem.replace('.sdf', '') + '.ply'
            path_to_latent[path] = latent_numpy

    Log.info('Latent code evaluation done. count = %s', len(path_to_latent))
    return path_to_latent


def encode_texts(texts, t5_cache_path, t5_model_name, t5_batch_size, device, t5_max_sentence_length):
    Log.info('Loading T5 model')
    tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix())
    model = T5EncoderModel.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix()).to(device)

    texts = list(texts)
    text_to_e_text = {}
    for start in tqdm(range(0, len(texts), t5_batch_size), desc="Encoding sentences"):
        slice_texts = texts[start:min(start + t5_batch_size, len(texts))]
        input_ids = tokenizer(
            slice_texts,
            return_tensors="pt",
            padding='max_length',
            max_length=t5_max_sentence_length,
            truncation=True,
        ).input_ids
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids)
        encoded_text = outputs.last_hidden_state.detach().cpu().numpy()

        for idx, e_text in enumerate(encoded_text):
            text_to_e_text[slice_texts[idx]] = e_text

    return text_to_e_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate diffusion dataset.')
    parser.add_argument('--sdf_ckpt_path', type=str, required=True, help='Checkpoint of SDF model.')
    parser.add_argument('--dataset_dir', type=Path, default=Path('../datasets'))
    parser.add_argument('--mesh_info_path', type=Path, default=Path('../datasets/1_preprocessed_info'))
    parser.add_argument('--output_path', type=Path, default=Path('../datasets/2.1_text_n_latentcode'))
    parser.add_argument('--t5_cache_path', type=Path, default=Path('../../cache/t5_cache'))
    parser.add_argument('--t5_model_name', type=str, default='google-t5/t5-large')
    parser.add_argument('--t5_batch_size', type=int, default=16)
    parser.add_argument('--t5_max_sentence_length', type=int, default=16)
    parser.add_argument('--use_dummy_text_embedding', action='store_true')
    parser.add_argument('--reset_output', action='store_true')
    parser.add_argument('--latent_batch_size', type=int, default=20)
    parser.add_argument('--latent_num_workers', type=int, default=5)
    parser.add_argument('--latent_samples_per_mesh', type=int, default=16000)
    parser.add_argument('--latent_pc_size', type=int, default=4096)
    parser.add_argument('--latent_uniform_sample_ratio', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_ckpt_path = Path(args.sdf_ckpt_path)

    gensdf_model = determine_latentcode_encoder(best_ckpt_path, device)

    output_path = args.output_path
    if args.reset_output:
        shutil.rmtree(str(output_path), ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)
    mesh_info_path = args.mesh_info_path

    t5_cache_path = args.t5_cache_path
    t5_cache_path.mkdir(exist_ok=True, parents=True)

    Log.info('Evaluating latent codes')
    path_to_latent = evaluate_latent_codes(
        gensdf_model,
        args.dataset_dir,
        device,
        args.latent_batch_size,
        args.latent_num_workers,
        args.latent_samples_per_mesh,
        args.latent_pc_size,
        args.latent_uniform_sample_ratio,
    )

    texts = set()
    for shape_json_path in sorted(mesh_info_path.glob('*.json')):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = camel_to_snake(shape_json['meta']['catecory'])
        for part_info in shape_json['part']:
            texts.add(f"{shape_name}, {part_info['name']}")

    print(texts)

    if args.use_dummy_text_embedding:
        Log.warning('Using dummy text embeddings. This is only for smoke tests.')
        rng = np.random.RandomState(42)
        text_to_e_text = {
            t: rng.randn(args.t5_max_sentence_length, 1024).astype(np.float32)
            for t in texts
        }
    else:
        text_to_e_text = encode_texts(
            texts,
            t5_cache_path,
            args.t5_model_name,
            args.t5_batch_size,
            device,
            args.t5_max_sentence_length,
        )

    failed = []
    success = []
    for shape_json_path in sorted(mesh_info_path.glob('*.json')):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = camel_to_snake(shape_json['meta']['catecory'])
        for part_info in shape_json['part']:
            mesh_name = Path(part_info['mesh']).stem
            if path_to_latent.get(part_info['mesh']) is None:
                Log.warning(f"Latent code for {mesh_name} not found")
                failed.append(mesh_name)
                continue
            np.savez(
                output_path / f'{mesh_name}.npz',
                latent_code=path_to_latent[part_info['mesh']],
                text=text_to_e_text[f"{shape_name}, {part_info['name']}"],
                text_label=np.array(f"{shape_name}, {part_info['name']}", dtype=str),
            )
            success.append(mesh_name)

    with open(output_path / 'meta.json', 'w') as f:
        json.dump({
            'ckpt': best_ckpt_path.as_posix(),
            'dataset_dir': args.dataset_dir.as_posix(),
            'mesh_info_path': mesh_info_path.as_posix(),
            'failed': failed,
            'success': success,
        }, f, indent=2)

    Log.info('Failed to find latent code for %s', failed)
    Log.info('failed count = %s', len(failed))
    Log.info('success count = %s', len(success))
    Log.info('Done')
