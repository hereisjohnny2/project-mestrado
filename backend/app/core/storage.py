"""Filesystem layout helper.

Mirrors the layout described in the plan (§4.2):

    storage/{project}/images/
    storage/{project}/masks/
    storage/{project}/datasets/
    storage/{project}/models/
    storage/{project}/runs/
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import get_settings


class ProjectStorage:
    def __init__(self, project_slug: str):
        self.root = get_settings().storage_dir / project_slug
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.datasets = self.root / "datasets"
        self.models = self.root / "models"
        self.runs = self.root / "runs"
        for d in (self.images, self.masks, self.datasets, self.models, self.runs):
            d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
