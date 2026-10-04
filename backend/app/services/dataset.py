"""Dataset generation: turns a project's annotation masks into the ``.dat``
format the legacy training core expects (plan §4.3 / §4.4,
``POST /projects/{id}/datasets``).

Masks are single-channel PNGs whose pixel value is the class index (``0`` =
unannotated, ``1..K`` = the project's classes) — see
``frontend/src/features/annotation/AnnotationCanvas.tsx``. For every
annotated pixel we look up the RGB of the same coordinate in the source
image and emit a ``R\\tG\\tB\\tLabel`` row, exactly the tabulation
``legacy/rock-image-annotation`` produced by hand. The label is the class
name, and the class list is stored with the dataset so the labels stay
interpretable.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from ..core.storage import ProjectStorage, get_settings, sha256_file
from ..db import models
from .classes import classes_from_labels


class EmptyDatasetError(ValueError):
    pass


class InvalidDatasetError(ValueError):
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

    classes = [dict(c) for c in project.classes]
    counts = {c["name"]: 0 for c in classes}
    with open(dest, "w") as f:
        for image in annotated:
            rgb = np.asarray(PILImage.open(storage_dir / image.path).convert("RGB"))
            mask = np.asarray(PILImage.open(storage_dir / image.annotation.mask_path).convert("L"))

            for cls in classes:
                ys, xs = np.where(mask == cls["index"])
                if ys.size == 0:
                    continue
                counts[cls["name"]] += int(ys.size)
                for r, g, b in rgb[ys, xs]:
                    f.write(f"{r}\t{g}\t{b}\t{cls['name']}\n")

    n_pixels = sum(counts.values())
    if n_pixels == 0:
        dest.unlink(missing_ok=True)
        raise EmptyDatasetError("no annotated pixels found")

    return {
        "id": dataset_id,
        "path": str(dest.relative_to(storage_dir)),
        "n_pixels": n_pixels,
        "classes": classes,
        "class_counts": counts,
        "sha256": sha256_file(dest),
    }


def dataset_absolute_path(dataset: models.Dataset) -> Path:
    return get_settings().storage_dir / dataset.path


def import_dataset(source, storage: ProjectStorage) -> dict:
    """Stores an uploaded legacy ``.dat`` (``R\\tG\\tB\\tLabel`` per line) as a
    dataset. ``source`` is a binary file object. Every line is validated —
    the legacy loader would crash mid-training on a malformed one. The
    classes are the distinct labels found, ``Poro`` first (see
    ``classes_from_labels``)."""
    storage_dir = get_settings().storage_dir
    dataset_id = str(uuid.uuid4())
    dest = storage.datasets / f"{dataset_id}.dat"

    counts: dict[str, int] = {}
    try:
        with open(dest, "w") as out:
            for number, raw in enumerate(io.TextIOWrapper(source, encoding="utf-8", errors="replace"), start=1):
                line = raw.rstrip("\r\n")
                fields = line.split("\t")
                try:
                    if len(fields) != 4:
                        raise ValueError
                    rgb = [int(v) for v in fields[:3]]
                    if not all(0 <= v <= 255 for v in rgb) or not fields[3]:
                        raise ValueError
                except ValueError:
                    raise InvalidDatasetError(
                        f"linha {number} inválida: esperado 'R<TAB>G<TAB>B<TAB>Rótulo' com R,G,B entre 0 e 255"
                    ) from None
                counts[fields[3]] = counts.get(fields[3], 0) + 1
                out.write(line + "\n")
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    if not counts:
        dest.unlink(missing_ok=True)
        raise InvalidDatasetError("arquivo vazio")

    classes = classes_from_labels(list(counts))
    return {
        "id": dataset_id,
        "path": str(dest.relative_to(storage_dir)),
        "n_pixels": sum(counts.values()),
        "classes": classes,
        "class_counts": {c["name"]: counts[c["name"]] for c in classes},
        "sha256": sha256_file(dest),
    }
