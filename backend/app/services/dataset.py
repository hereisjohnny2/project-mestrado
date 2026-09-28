"""Dataset generation: turns a project's annotation masks into the ``.dat``
format the legacy training core expects (plan §4.3 / §4.4,
``POST /projects/{id}/datasets``).

Masks are single-channel PNGs with pixel values ``0`` (unannotated),
``1`` (poro) or ``2`` (solido) — see
``frontend/src/features/annotation/AnnotationCanvas.tsx``. For every
annotated pixel we look up the RGB of the same coordinate in the source
image and emit a ``R\\tG\\tB\\tLabel`` row, exactly the tabulation
``legacy/rock-image-annotation`` produced by hand.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from ..core.storage import ProjectStorage, get_settings, sha256_file
from ..db import models

LABEL_NAMES = {1: "Poro", 2: "Solido"}


class EmptyDatasetError(ValueError):
    pass


def generate_dataset(project: models.Project, storage: ProjectStorage) -> dict:
    """Writes a new ``.dat`` file from every annotated image in ``project``.

    Returns the fields needed to build a ``Dataset`` row. Raises
    ``EmptyDatasetError`` if no image has an annotation, or no annotated
    pixel exists at all.
    """
    annotated = [image for image in project.images if image.annotation is not None]
    if not annotated:
        raise EmptyDatasetError("project has no annotated images")

    storage_dir = get_settings().storage_dir
    dataset_id = str(uuid.uuid4())
    dest = storage.datasets / f"{dataset_id}.dat"

    n_pore = 0
    n_solid = 0
    with open(dest, "w") as f:
        for image in annotated:
            rgb = np.asarray(PILImage.open(storage_dir / image.path).convert("RGB"))
            mask = np.asarray(PILImage.open(storage_dir / image.annotation.mask_path).convert("L"))

            for label_value, label_name in LABEL_NAMES.items():
                ys, xs = np.where(mask == label_value)
                if ys.size == 0:
                    continue
                if label_value == 1:
                    n_pore += int(ys.size)
                else:
                    n_solid += int(ys.size)
                pixels = rgb[ys, xs]
                for r, g, b in pixels:
                    f.write(f"{r}\t{g}\t{b}\t{label_name}\n")

    n_pixels = n_pore + n_solid
    if n_pixels == 0:
        dest.unlink(missing_ok=True)
        raise EmptyDatasetError("no annotated pixels found")

    return {
        "id": dataset_id,
        "path": str(dest.relative_to(storage_dir)),
        "n_pixels": n_pixels,
        "n_pore": n_pore,
        "n_solid": n_solid,
        "sha256": sha256_file(dest),
    }


def dataset_absolute_path(dataset: models.Dataset) -> Path:
    return get_settings().storage_dir / dataset.path
