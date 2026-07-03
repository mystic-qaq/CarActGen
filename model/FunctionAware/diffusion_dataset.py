from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import trange

from .functions import FUNCTION_TO_ID, infer_function_label


class FunctionAwareDiffusionDataset(Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        self.files = sorted(p for p in dataset_path.glob("*.npz") if p.name != "meta.npz")
        self.meta = json.loads((dataset_path / "meta.json").read_text())

        self.data = [None] * len(self.files)
        for idx in trange(len(self.files), desc="Loading and cache function-aware diffusion data."):
            data = np.load(self.files[idx], allow_pickle=True)
            function_id = data["function_id"].astype(np.int64) if "function_id" in data.files else np.array(0, dtype=np.int64)
            self.data[idx] = (
                data["text"].astype(np.float32),
                data["latent_code"].astype(np.float32),
                function_id,
            )

    def get_gensdf_ckpt_path(self):
        return self.meta["ckpt"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.data[index]

