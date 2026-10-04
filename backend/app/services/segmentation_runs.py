"""Background execution of batch segmentation runs (plan §4.4, Fase 3).

Same threading approach as ``training_jobs``: single-user, single-process.
Unlike training, results are persisted per image as they are produced, so a
finished run survives restarts; only the in-flight ``total`` lives in memory
(falling back to the number of stored results for runs from a previous
process).
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from ..core.config import get_settings
from ..core.storage import ProjectStorage
from ..db import models
from ..db.session import get_session_factory
from ..ml.artifacts import load_model
from ..ml.inference import BINARY_PALETTE, apply_binarization, class_fractions, save_class_map, save_overlay
from .classes import colors_by_index, output_mapping, pore_class_name

TOTALS: dict[str, int] = {}


def run_progress(run: models.Run) -> tuple[int, int]:
    done = len(run.results)
    return done, TOTALS.get(run.id, done)


def start_run(run_id: str, image_ids: list[str]) -> None:
    TOTALS[run_id] = len(image_ids)
    threading.Thread(target=_execute, args=(run_id, image_ids), daemon=True).start()


def _execute(run_id: str, image_ids: list[str]) -> None:
    db = get_session_factory()()
    try:
        run = db.get(models.Run, run_id)
        run.status = models.RunStatus.RUNNING
        db.commit()
        try:
            _process(db, run, image_ids)
            run.status = models.RunStatus.DONE
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as run.error
            db.rollback()
            run = db.get(models.Run, run_id)
            run.error = str(exc)
            run.status = models.RunStatus.FAILED
        run.finished_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _process(db, run: models.Run, image_ids: list[str]) -> None:
    storage_dir = get_settings().storage_dir
    model = run.model
    index_to_name, overlay_colors = output_mapping(model)
    net = load_model(
        storage_dir / model.pt_path,
        architecture=model.architecture,
        n_classes=len(index_to_name),
        hidden_width=model.config.get("hidden_width"),
    )
    storage = ProjectStorage(model.dataset.project_id)
    out_dir = storage.runs / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    # The binarized image stays black/white for "pore vs. rest" models (the
    # legacy output); multi-class maps are rendered with the class colors.
    palette = BINARY_PALETTE if model.architecture == "1.0.0" else {0: (0, 0, 0), **colors_by_index(model.dataset.classes)}
    pore_name = pore_class_name(model.dataset.classes)

    for image_id in image_ids:
        image = db.get(models.Image, image_id)
        if image is None:
            continue
        src = str(storage_dir / image.path)
        class_map, elapsed_ms = apply_binarization(src, net, model.architecture)
        prefix = str(out_dir / image.id)
        bin_path = save_class_map(class_map, prefix, palette)
        overlay_path = save_overlay(src, class_map, prefix, overlay_colors)
        fractions = class_fractions(class_map, index_to_name)
        db.add(
            models.Result(
                run_id=run.id,
                image_id=image.id,
                porosity=fractions.get(pore_name, 0.0),
                class_fractions=fractions,
                time_ms=elapsed_ms,
                bin_path=_rel(bin_path, storage_dir),
                overlay_path=_rel(overlay_path, storage_dir),
            )
        )
        db.commit()
        db.refresh(run)


def _rel(path: str, storage_dir) -> str:
    return str(Path(path).relative_to(storage_dir))
