"""Per-project class lists: validation and the helpers that turn the JSON
stored on ``Project``/``Dataset`` into names, colors and index mappings.

Masks and model outputs refer to classes by index, so a class list is an
ordered sequence ``1..K`` with unique names and a hex color each.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image as PILImage

from ..core.config import get_settings
from ..db import models

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_NAME_LENGTH = 40
MAX_CLASSES = 32

# Colors handed to classes that arrive without one (legacy .dat imports).
PALETTE = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2",
    "#ca8a04", "#db2777", "#4b5563", "#65a30d", "#7c3aed", "#f97316",
]


class InvalidClassesError(ValueError):
    pass


class ClassInUseError(ValueError):
    pass


def validate_classes(raw: list[dict]) -> list[dict]:
    if not raw:
        raise InvalidClassesError("o projeto precisa de ao menos uma classe")
    if len(raw) > MAX_CLASSES:
        raise InvalidClassesError(f"no máximo {MAX_CLASSES} classes")
    out: list[dict] = []
    seen: set[str] = set()
    for position, item in enumerate(raw, start=1):
        index = item.get("index")
        if index != position:
            raise InvalidClassesError("os índices das classes devem ser 1, 2, 3… em ordem")
        name = str(item.get("name") or "").strip()
        if not name:
            raise InvalidClassesError(f"classe {position}: nome vazio")
        if len(name) > MAX_NAME_LENGTH:
            raise InvalidClassesError(f"classe {position}: nome com mais de {MAX_NAME_LENGTH} caracteres")
        if name.lower() in seen:
            raise InvalidClassesError(f"nome de classe repetido: {name}")
        seen.add(name.lower())
        color = str(item.get("color") or "").strip().lower()
        if not _HEX_COLOR.match(color):
            raise InvalidClassesError(f"classe {name}: cor deve ser #rrggbb")
        out.append({"index": position, "name": name, "color": color})
    return out


def class_names(classes: list[dict]) -> list[str]:
    """Names ordered by index (index 1 first)."""
    return [c["name"] for c in sorted(classes, key=lambda c: c["index"])]


def color_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def colors_by_index(classes: list[dict]) -> dict[int, tuple[int, int, int]]:
    return {c["index"]: color_rgb(c["color"]) for c in classes}


def pore_class_name(classes: list[dict]) -> str:
    """Index 1 is, by convention, the class whose fraction is the porosity."""
    return class_names(classes)[0]


def binary_class_names(classes: list[dict]) -> list[str]:
    """Output names of a "pore vs. rest" (architecture 1.0.0) model: index 0
    is everything that is not the pore class."""
    names = class_names(classes)
    rest = names[1] if len(names) == 2 else "Outros"
    return [rest, names[0]]


def output_class_names(model: models.MLModel) -> list[str]:
    """Names of the model's outputs, in output order. Models published
    before class lists existed carry none in their config."""
    return list(model.config.get("class_names") or binary_class_names(model.dataset.classes))


def output_mapping(model: models.MLModel) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    """How the class map a model produces relates to the project's classes:
    ``(index -> output name, index -> overlay color)``.

    Architecture 1.0.0 is "pore vs. rest": index 1 is the project's pore
    class (index 1), index 0 is the rest and gets no overlay color. Other
    architectures write the project's class indices directly."""
    names = output_class_names(model)
    classes = sorted(model.dataset.classes, key=lambda c: c["index"])
    if model.architecture == "1.0.0":
        return {0: names[0], 1: names[1]}, {1: color_rgb(classes[0]["color"])}
    return {c["index"]: names[c["index"] - 1] for c in classes}, colors_by_index(classes)


def classes_from_labels(labels: list[str]) -> list[dict]:
    """Class list for an imported ``.dat``: distinct labels in order of
    appearance, with the legacy ``Poro`` label pinned to index 1 so the
    porosity convention holds."""
    distinct = list(dict.fromkeys(labels))
    if "Poro" in distinct:
        distinct.remove("Poro")
        distinct.insert(0, "Poro")
    defaults = {c["name"]: c["color"] for c in models.DEFAULT_CLASSES}
    defaults["Solido"] = defaults["Sólido"]
    return [
        {"index": i, "name": name, "color": defaults.get(name, PALETTE[(i - 1) % len(PALETTE)])}
        for i, name in enumerate(distinct, start=1)
    ]


def indices_in_use(project: models.Project) -> set[int]:
    storage_dir = get_settings().storage_dir
    used: set[int] = set()
    for image in project.images:
        if image.annotation is None:
            continue
        mask = np.asarray(PILImage.open(storage_dir / image.annotation.mask_path).convert("L"))
        used.update(int(v) for v in np.unique(mask) if v != 0)
    return used


def replace_classes(project: models.Project, raw: list[dict]) -> list[dict]:
    """Validates and applies a full class list. Removing a class is only
    allowed when no mask in the project uses its index."""
    classes = validate_classes(raw)
    kept = {c["index"] for c in classes}
    removed = {c["index"] for c in project.classes} - kept
    if removed:
        in_use = removed & indices_in_use(project)
        if in_use:
            names = {c["index"]: c["name"] for c in project.classes}
            raise ClassInUseError(
                "classe em uso por anotações: " + ", ".join(names[i] for i in sorted(in_use))
            )
    project.classes = classes
    return classes
