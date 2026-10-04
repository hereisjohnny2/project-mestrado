"""Model artifact persistence.

The legacy CLI (``trainer.py``) saved only a bare ``state_dict`` as
``<dataset>-nn-model.pt`` — no hyperparameters, no metrics, no dataset
provenance (plan §1.3, item 3). Here every trained model gets:

- ``<name>.pt``            — ``state_dict``, loadable exactly like before
  (``model.load_state_dict(torch.load(path))``), so legacy tooling and the
  parity test can still read it directly.
- ``<name>.json``          — hyperparameters, dataset hash, metrics, seed,
  timestamp, format version.
- ``<name>-scripted.pt``   — TorchScript export, equivalent to
  ``legacy/rock-nn/model_serializer.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .architectures import build_model, get_architecture, resolve_hidden_width
from .model import RockNetModel
from .training import TrainingResult

# 1: bare legacy port. 2: adds architecture / hidden_width / class_names /
# feature_names (format-1 files are always architecture 1.0.0).
ARTIFACT_FORMAT_VERSION = 2


def save_model(result: TrainingResult, output_dir: str | Path, name: str, dataset_sha256: str | None = None) -> dict:
    """Writes the .pt / .json / -scripted.pt triple and returns their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / f"{name}.pt"
    json_path = output_dir / f"{name}.json"
    scripted_path = output_dir / f"{name}-scripted.pt"

    torch.save(result.model.state_dict(), pt_path)

    scripted = torch.jit.script(result.model.eval())
    scripted.save(str(scripted_path))

    architecture = result.config.architecture
    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": time.time(),
        "dataset_sha256": dataset_sha256,
        "architecture": architecture,
        "hidden_width": resolve_hidden_width(architecture, result.config.hidden_width),
        "class_names": result.class_names,
        "feature_names": get_architecture(architecture).feature_names,
        "config": asdict(result.config),
        "metrics": {
            "accuracy": result.accuracy,
            "loss_curve": result.loss_curve,
            "confusion_matrix": result.confusion_matrix,
            "per_class": result.per_class,
            "duration_ms": result.duration_ms,
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2))

    return {"pt_path": str(pt_path), "json_path": str(json_path), "scripted_path": str(scripted_path)}


def save_imported_model(
    net: RockNetModel,
    output_dir: str | Path,
    name: str,
    metrics: dict,
    dataset_sha256: str | None = None,
    source_filename: str | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Same .pt / .json / -scripted.pt triple as ``save_model``, for a
    ``state_dict`` trained elsewhere (the legacy CLI, so always architecture
    1.0.0). There is no training config to record, so ``config`` only marks
    the model as imported."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / f"{name}.pt"
    json_path = output_dir / f"{name}.json"
    scripted_path = output_dir / f"{name}-scripted.pt"

    torch.save(net.state_dict(), pt_path)
    torch.jit.script(net.eval()).save(str(scripted_path))

    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": time.time(),
        "dataset_sha256": dataset_sha256,
        "architecture": "1.0.0",
        "hidden_width": None,
        "class_names": class_names,
        "feature_names": get_architecture("1.0.0").feature_names,
        "config": {"imported": True, "source_filename": source_filename},
        "metrics": metrics,
    }
    json_path.write_text(json.dumps(metadata, indent=2))

    return {"pt_path": str(pt_path), "json_path": str(json_path), "scripted_path": str(scripted_path)}


def load_state_dict_safely(path: str | Path) -> RockNetModel:
    """Loads an uploaded ``.pt`` into a fresh ``RockNetModel``.

    ``weights_only=True`` keeps ``torch.load`` from unpickling arbitrary
    objects out of an untrusted upload. Raises ``ValueError`` if the file is
    not a ``state_dict`` of this architecture (e.g. a TorchScript export).
    """
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = RockNetModel()
        model.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 - any failure means "not a compatible state_dict"
        raise ValueError(
            "arquivo não é um state_dict compatível com RockNetModel (3→4→4→4→2); "
            "modelos TorchScript (-scripted.pt) não podem ser importados"
        ) from exc
    model.eval()
    return model


def load_model(
    pt_path: str | Path,
    architecture: str = "1.0.0",
    n_classes: int = 2,
    hidden_width: int | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Loads a ``state_dict`` into a fresh model of the given architecture.
    For 1.0.0 the .pt format is the legacy CLI's, so both are interchangeable."""
    model = build_model(architecture, n_classes=n_classes, hidden_width=hidden_width)
    state_dict = torch.load(pt_path, map_location=device or "cpu")
    model.load_state_dict(state_dict)
    model.eval()
    if device and device != "cpu":
        model.to(device)
    return model


def load_metadata(json_path: str | Path) -> dict:
    return json.loads(Path(json_path).read_text())
