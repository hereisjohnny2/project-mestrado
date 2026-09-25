"""Image ingestion: convert an uploaded file to RGB PNG and land it in the
project's storage tree (plan §4.4: ``POST /projects/{id}/images``)."""

from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image as PILImage

from ..core.storage import ProjectStorage, sha256_file


class UnsupportedImageError(ValueError):
    pass


def ingest_image(storage: ProjectStorage, filename: str, raw: bytes) -> dict:
    """Decodes ``raw``, converts it to RGB, and saves it as
    ``{image_id}.png`` under the project's ``images/`` directory.

    Returns the fields needed to build an ``Image`` row.
    """
    import io

    try:
        img = PILImage.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # pragma: no cover - PIL raises many subclasses
        raise UnsupportedImageError(f"could not decode image {filename!r}: {exc}") from exc

    rgb = img.convert("RGB")
    image_id = str(uuid.uuid4())
    dest = storage.images / f"{image_id}.png"
    rgb.save(dest, format="PNG")

    return {
        "id": image_id,
        "filename": filename,
        "path": str(dest.relative_to(storage.root.parent)),
        "width": rgb.width,
        "height": rgb.height,
        "sha256": sha256_file(dest),
    }


def mask_relative_path(storage: ProjectStorage, image_id: str) -> tuple[Path, str]:
    """Returns the absolute path to write the mask to, and its path relative
    to the storage root (what gets stored in ``Annotation.mask_path``)."""
    dest = storage.masks / f"{image_id}.png"
    return dest, str(dest.relative_to(storage.root.parent))
