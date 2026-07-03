import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

def center(bbx):
    return np.array([(bbx[0][i] + bbx[1][i]) / 2.0 for i in range(3)], dtype=np.float32)


def size(bbx):
    return np.array([(bbx[1][i] - bbx[0][i]) for i in range(3)], dtype=np.float32)


def collect_metrics(parts):
    part_map = {part['name']: part for part in parts}
    body_size = size(part_map['body_shell']['bbx'])
    wheel_names = [
        'wheel_front_left',
        'wheel_front_right',
        'wheel_rear_left',
        'wheel_rear_right',
    ]
    wheel_centers = {name: center(part_map[name]['bbx']) for name in wheel_names}
    wheel_size = size(part_map['wheel_front_left']['bbx'])

    front_x = (wheel_centers['wheel_front_left'][0] + wheel_centers['wheel_front_right'][0]) / 2.0
    rear_x = (wheel_centers['wheel_rear_left'][0] + wheel_centers['wheel_rear_right'][0]) / 2.0
    left_y = (wheel_centers['wheel_front_left'][1] + wheel_centers['wheel_rear_left'][1]) / 2.0
    right_y = (wheel_centers['wheel_front_right'][1] + wheel_centers['wheel_rear_right'][1]) / 2.0

    return {
        'body_length': float(body_size[0]),
        'body_width': float(body_size[1]),
        'body_height': float(body_size[2]),
        'wheelbase': float(rear_x - front_x),
        'track_width': float(abs(left_y - right_y)),
        'wheel_radius': float((wheel_size[0] + wheel_size[2]) / 4.0),
    }


def make_bucket_fn(values, labels):
    low, high = np.quantile(np.array(values, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])

    def bucket(value):
        if value <= low:
            return labels[0]
        if value <= high:
            return labels[1]
        return labels[2]

    return bucket


def format_texts(metrics, labelers):
    body_len_label = labelers['body_length'](metrics['body_length'])
    body_height_label = labelers['body_height'](metrics['body_height'])
    wheelbase_label = labelers['wheelbase'](metrics['wheelbase'])
    track_label = labelers['track_width'](metrics['track_width'])

    return [
        (
            f"A {body_height_label} articulated car with a {body_len_label} body, a "
            f"{wheelbase_label}, and a {track_label}. The object contains one body shell "
            f"and four wheel parts. The wheels are arranged at the front left, front right, "
            f"rear left, and rear right positions, and every wheel rotates with a revolute joint "
            f"around a shared lateral axis."
        ),
        (
            f"This articulated vehicle has one body shell and four wheels. It shows a "
            f"{body_len_label} silhouette, a {body_height_label}, and a {track_label}, with "
            f"the rear wheels placed behind a {wheelbase_label}."
        ),
        (
            f"A four-wheel articulated car with body length {metrics['body_length']:.2f} units, "
            f"body height {metrics['body_height']:.2f} units, wheelbase {metrics['wheelbase']:.2f}, "
            f"track width {metrics['track_width']:.2f}, and wheel radius about "
            f"{metrics['wheel_radius']:.2f} units."
        ),
        (
            f"Keywords: articulated car, single body shell, four revolute wheels, {body_height_label}, "
            f"{body_len_label}, {wheelbase_label}, {track_label}, symmetric wheel layout."
        ),
        "An articulated car with one body shell and four revolute wheel parts.",
    ]


def main(input_dir: Path, output_dir: Path):
    shape_paths = sorted(input_dir.glob('*.json'))
    if not shape_paths:
        raise FileNotFoundError(f'No shape json found in {input_dir}')

    all_metrics = []
    shape_to_parts = {}
    for shape_path in shape_paths:
        shape_json = json.loads(shape_path.read_text())
        parts = shape_json['part']
        shape_to_parts[shape_path.stem] = parts
        all_metrics.append(collect_metrics(parts))

    labelers = {
        'body_length': make_bucket_fn(
            [item['body_length'] for item in all_metrics],
            ['compact-length', 'mid-length', 'long-body'],
        ),
        'body_height': make_bucket_fn(
            [item['body_height'] for item in all_metrics],
            ['low-profile', 'mid-height', 'tall-roof'],
        ),
        'wheelbase': make_bucket_fn(
            [item['wheelbase'] for item in all_metrics],
            ['short-wheelbase', 'balanced-wheelbase', 'long-wheelbase'],
        ),
        'track_width': make_bucket_fn(
            [item['track_width'] for item in all_metrics],
            ['narrow-track', 'balanced-track', 'wide-track'],
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for shape_name, parts in tqdm(shape_to_parts.items(), desc='Writing car descriptions'):
        metrics = collect_metrics(parts)
        texts = format_texts(metrics, labelers)

        shape_output_dir = output_dir / shape_name
        shape_output_dir.mkdir(parents=True, exist_ok=True)
        for idx, text in enumerate(texts):
            (shape_output_dir / f'{idx}.txt').write_text(text)

    (output_dir / 'meta.json').write_text(json.dumps({
        'count': len(shape_to_parts),
        'description_per_shape': 5,
        'source': 'programmatic_car_descriptions',
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate car text descriptions.')
    parser.add_argument('--input_dir', type=Path, default=Path('../datasets/1_preprocessed_info'))
    parser.add_argument('--output_dir', type=Path, default=Path('../datasets/3_text_condition'))
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
