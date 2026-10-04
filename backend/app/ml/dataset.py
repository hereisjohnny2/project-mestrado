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
    pore_label: str = "Poro",
) -> tuple[DataLoader, DataLoader]:
    dataset = CustomDataset(load_data_from_file(data, pore_label))
    train_dataset, test_dataset = split_dataset(dataset, ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_rows(file_name: str | Path) -> list[tuple[int, int, int, str]]:
    """Every ``.dat`` row with its label kept as text — the multi-class view
    of the same file ``load_data_from_file`` reads as pore / not-pore."""
    rows = []
    with open(file_name, "r") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append((int(fields[0]), int(fields[1]), int(fields[2]), fields[3]))
    return rows


def dataset_stats(data: str | Path, class_names: list[str]) -> dict:
    """Class balance for a ``.dat`` file — used by the dataset screen."""
    rows = load_rows(data)
    counts = {name: 0 for name in class_names}
    for *_, label in rows:
        if label in counts:
            counts[label] += 1
    n_total = len(rows)
    return {
        "n_pixels": n_total,
        "class_counts": counts,
        "class_ratios": {name: (n / n_total if n_total else 0.0) for name, n in counts.items()},
    }


def dataset_histogram(data: str | Path, class_names: list[str], bins: int = 32) -> dict:
    """Per-channel, per-class RGB histogram (plan §5.2: shows whether the
    classes are separable in color space, i.e. what the MLP actually sees).
    0-255 range, fixed bin count regardless of dataset size."""
    import numpy as np

    rows = load_rows(data)
    rgb = np.array([row[:3] for row in rows], dtype=np.int64).reshape(-1, 3)
    labels = np.array([row[3] for row in rows])
    edges = np.linspace(0, 255, bins + 1)

    def channel_hist(mask: np.ndarray, channel: int) -> list[int]:
        counts, _ = np.histogram(rgb[mask, channel], bins=edges)
        return counts.tolist()

    classes = {}
    for name in class_names:
        mask = labels == name
        classes[name] = {"r": channel_hist(mask, 0), "g": channel_hist(mask, 1), "b": channel_hist(mask, 2)}
    return {"bin_edges": edges.tolist(), "classes": classes}
