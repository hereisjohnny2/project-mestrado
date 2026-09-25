"""Dataset loading, ported from ``legacy/rock-nn/custom_dataset.py`` and
``legacy/rock-nn/utils/dataset.py``.

Behaviour kept identical on purpose (see ``backend/tests/test_parity.py``):

- ``.dat`` files are tab-separated ``R\\tG\\tB\\tLabel`` lines.
- The label ``"Poro"`` maps to class ``1``, anything else to class ``0``.
- RGB values are fed to the network raw (0-255 floats), no normalization.
- The train/test split ratio defaults to 0.8 / 0.2, using
  ``torch.utils.data.random_split`` on the *global* RNG so that seeding is
  done by the caller (``torch.manual_seed``) exactly like the legacy CLI
  would if it were seeded.
- Batch size defaults to 16, train loader shuffled, test loader not.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """Same shape as legacy ``CustomDataset``: (R, G, B) -> label."""

    def __init__(self, content_data: list[list[int]]):
        self.content_data = content_data

    def __len__(self) -> int:
        return len(self.content_data)

    def __getitem__(self, index: int):
        data = self.content_data[index]
        rgb = torch.Tensor(data[0:3])
        label = data[3]
        return rgb, label


def load_data_from_file(file_name: str | Path, pore_label: str = "Poro") -> list[list[int]]:
    content = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            mod_line = line.strip("\n").split("\t")
            rgb = [int(i) for i in mod_line[0:3]]
            label = 1 if mod_line[3] == pore_label else 0
            rgb.append(label)
            content.append(rgb)
    return content


def split_dataset(dataset: Dataset, ratio: float = 0.8):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def create_dataloaders(
    data: str | Path,
    ratio: float = 0.8,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    dataset = CustomDataset(load_data_from_file(data))
    train_dataset, test_dataset = split_dataset(dataset, ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def dataset_stats(data: str | Path, pore_label: str = "Poro") -> dict:
    """Class balance for a ``.dat`` file — used by the dataset screen."""
    content = load_data_from_file(data, pore_label)
    n_pore = sum(row[3] for row in content)
    n_total = len(content)
    return {
        "n_pixels": n_total,
        "n_pore": n_pore,
        "n_solid": n_total - n_pore,
        "pore_ratio": (n_pore / n_total) if n_total else 0.0,
    }
