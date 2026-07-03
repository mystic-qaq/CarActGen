from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Tuple


FUNCTION_VOCAB = [
    "static_root",
    "static_part",
    "rotating_support",
    "rotating_part",
    "hinged_panel",
    "sliding_part",
    "handle",
    "support",
]

FUNCTION_TO_ID = {name: idx for idx, name in enumerate(FUNCTION_VOCAB)}


def _has_motion(part_info: dict) -> bool:
    limit = part_info.get("limit") or [0.0, 0.0, 0.0, 0.0]
    direction = part_info.get("joint_data_direction") or [0.0, 0.0, 0.0]
    return max(abs(float(x)) for x in limit + direction) > 1e-6


def infer_function_label(category: str, part_info: dict) -> str:
    """Map category-specific part names to a reusable functional role."""
    name = str(part_info.get("name", "")).lower()
    is_root = int(part_info.get("dfn_fa", -1)) == 0
    moving = _has_motion(part_info)

    if is_root or any(k in name for k in ("body", "shell", "case", "base_frame")):
        return "static_root"
    if any(k in name for k in ("wheel", "tire", "tyre", "rim", "caster")):
        return "rotating_support"
    if any(k in name for k in ("door", "lid", "flap", "hinge")):
        return "hinged_panel" if moving else "static_part"
    if any(k in name for k in ("drawer", "slider", "slide")):
        return "sliding_part" if moving else "static_part"
    if any(k in name for k in ("handle", "knob", "grip", "pull")):
        return "handle"
    if any(k in name for k in ("leg", "stand", "support", "foot")):
        return "support"
    if moving:
        return "rotating_part"
    return "static_part"


def function_id(label: str) -> int:
    return FUNCTION_TO_ID.get(label, FUNCTION_TO_ID["static_part"])


def load_function_map(mesh_info_dir: Path) -> Dict[str, Tuple[int, str]]:
    mesh_info_dir = Path(mesh_info_dir)
    result: Dict[str, Tuple[int, str]] = {}
    if not mesh_info_dir.exists():
        return result

    for shape_json_path in sorted(mesh_info_dir.glob("*.json")):
        shape_json = json.loads(shape_json_path.read_text())
        category = shape_json.get("meta", {}).get("catecory", "")
        for part_info in shape_json.get("part", []):
            mesh_stem = Path(part_info["mesh"]).stem
            label = infer_function_label(category, part_info)
            result[mesh_stem] = (function_id(label), label)
    return result

