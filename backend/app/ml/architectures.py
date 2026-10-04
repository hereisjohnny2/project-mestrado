"""The model architectures a user can train, and what each one is.

This is the single description of a version — the backend builds models
from it and the UI shows its text in the "how it works" panel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn

from .features import FEATURE_NAMES_V2
from .model import RockNetModel, RockNetV2


@dataclass(frozen=True)
class Architecture:
    id: str
    title: str
    description: str
    feature_names: list[str]
    input_scale: str
    supports_multiclass: bool
    default_hidden_width: int | None


ARCHITECTURES: dict[str, Architecture] = {
    "1.0.0": Architecture(
        id="1.0.0",
        title="RGB por pixel (dissertação)",
        description=(
            "Rede original da dissertação, mantida bit a bit igual ao código legado. "
            "Cada pixel entra como três valores RGB crus (0–255) e passa por um MLP "
            "3→4→4→4→2. É sempre binária: a saída 1 é a classe de índice 1 do projeto "
            "(poro) e a saída 0 é todo o resto — em projetos com mais de duas classes "
            "ela treina 'poro vs. resto'."
        ),
        feature_names=["R", "G", "B"],
        input_scale="0–255 cru",
        supports_multiclass=False,
        default_hidden_width=None,
    ),
    "2.0.0": Architecture(
        id="2.0.0",
        title="Atributos de cor por pixel",
        description=(
            "Cada pixel entra com oito atributos derivados da sua cor, todos em [0, 1]: "
            "R, G, B normalizados; saturação (S) e brilho (V) do HSV; luminosidade (L) e "
            "cromaticidade (a, b) do Lab. O matiz (H) fica de fora por ser circular e "
            "ruidoso em cores pouco saturadas. A rede é um MLP 8→W→W/2→K com largura W "
            "configurável e uma saída por classe do projeto, treinada com Adam e perda "
            "de log-verossimilhança negativa."
        ),
        feature_names=list(FEATURE_NAMES_V2),
        input_scale="[0, 1] normalizado",
        supports_multiclass=True,
        default_hidden_width=32,
    ),
}

DEFAULT_ARCHITECTURE = "1.0.0"


def get_architecture(architecture_id: str) -> Architecture:
    try:
        return ARCHITECTURES[architecture_id]
    except KeyError:
        raise ValueError(f"arquitetura desconhecida: {architecture_id}") from None


def resolve_hidden_width(architecture_id: str, hidden_width: int | None) -> int | None:
    arch = get_architecture(architecture_id)
    if arch.default_hidden_width is None:
        return None
    return hidden_width or arch.default_hidden_width


def layers_for(architecture_id: str, n_classes: int, hidden_width: int | None = None) -> list[int]:
    if architecture_id == "1.0.0":
        return [3, 4, 4, 4, 2]
    width = resolve_hidden_width(architecture_id, hidden_width)
    return [len(FEATURE_NAMES_V2), width, max(width // 2, 2), n_classes]


def build_model(architecture_id: str, n_classes: int, hidden_width: int | None = None) -> nn.Module:
    if architecture_id == "1.0.0":
        return RockNetModel()
    get_architecture(architecture_id)
    width = resolve_hidden_width(architecture_id, hidden_width)
    return RockNetV2(n_features=len(FEATURE_NAMES_V2), hidden_width=width, n_classes=n_classes)


def describe(architecture_id: str, n_classes: int = 2, hidden_width: int | None = None) -> dict:
    arch = get_architecture(architecture_id)
    return {
        **asdict(arch),
        "hidden_width": resolve_hidden_width(architecture_id, hidden_width),
        "layers": layers_for(architecture_id, n_classes, hidden_width),
    }
