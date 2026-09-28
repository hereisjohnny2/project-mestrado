"""Training jobs and published models (plan §4.4, Fase 2)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from ..core.storage import ProjectStorage, get_settings
from ..db import models
from ..db.session import get_db
from ..ml.artifacts import save_model
from ..ml.training import TrainingConfig
from ..schemas import ModelOut, PublishModelRequest, TrainingJobCreate, TrainingJobOut
from ..services.dataset import dataset_absolute_path
from ..services.training_jobs import get_job, start_job, wait_for_update

router = APIRouter()


@router.post("/training/jobs", response_model=TrainingJobOut, status_code=201)
def create_training_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> dict:
    dataset = db.get(models.Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")

    config = TrainingConfig(
        epochs=payload.epochs,
        learning_rate=payload.learning_rate,
        batch_size=payload.batch_size,
        split_ratio=payload.split_ratio,
        seed=payload.seed,
    )
    job = start_job(dataset.id, str(dataset_absolute_path(dataset)), config)
    return job.snapshot()


@router.get("/training/jobs/{job_id}", response_model=TrainingJobOut)
def get_training_job(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "training job not found")
    return job.snapshot()


@router.get("/training/jobs/{job_id}/stream")
def stream_training_job(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "training job not found")

    def events():
        last_epoch = -1
        last_status = ""
        while True:
            snapshot = job.snapshot()
            if snapshot["epoch"] != last_epoch or snapshot["status"] != last_status:
                last_epoch = snapshot["epoch"]
                last_status = snapshot["status"]
                yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["status"] in ("done", "failed"):
                return
            wait_for_update(job, last_epoch, last_status)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/training/jobs/{job_id}/publish", response_model=ModelOut, status_code=201)
def publish_model(job_id: str, payload: PublishModelRequest, db: Session = Depends(get_db)) -> models.MLModel:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "training job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, "training job has not finished successfully")

    dataset = db.get(models.Dataset, job.dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")

    storage = ProjectStorage(dataset.project_id)
    version = (
        db.query(models.MLModel)
        .join(models.Dataset)
        .filter(models.Dataset.project_id == dataset.project_id, models.MLModel.name == payload.name)
        .count()
        + 1
    )
    artifact_name = f"{payload.name}-v{version}"
    paths = save_model(job.result, storage.models, artifact_name, dataset_sha256=dataset.sha256)

    storage_dir = get_settings().storage_dir
    model = models.MLModel(
        dataset_id=dataset.id,
        name=payload.name,
        version=version,
        pt_path=str(Path(paths["pt_path"]).relative_to(storage_dir)),
        json_path=str(Path(paths["json_path"]).relative_to(storage_dir)),
        metrics={
            "accuracy": job.result.accuracy,
            "loss_curve": job.result.loss_curve,
            "confusion_matrix": job.result.confusion_matrix,
            "per_class": job.result.per_class,
            "duration_ms": job.result.duration_ms,
        },
        config={
            "epochs": job.result.config.epochs,
            "learning_rate": job.result.config.learning_rate,
            "batch_size": job.result.config.batch_size,
            "split_ratio": job.result.config.split_ratio,
            "seed": job.result.config.seed,
        },
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    job.model_id = model.id
    return model


@router.get("/projects/{project_id}/models", response_model=list[ModelOut])
def list_models(project_id: str, db: Session = Depends(get_db)) -> list[models.MLModel]:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return (
        db.query(models.MLModel)
        .join(models.Dataset)
        .filter(models.Dataset.project_id == project_id)
        .order_by(models.MLModel.created_at.desc())
        .all()
    )


@router.get("/models/{model_id}/download")
def download_model(model_id: str, fmt: str = "pt", db: Session = Depends(get_db)) -> FileResponse:
    model = db.get(models.MLModel, model_id)
    if model is None:
        raise HTTPException(404, "model not found")

    storage_dir = get_settings().storage_dir
    if fmt == "pt":
        path = storage_dir / model.pt_path
    elif fmt == "json":
        path = storage_dir / model.json_path
    elif fmt == "scripted":
        path = (storage_dir / model.pt_path).parent / f"{model.name}-v{model.version}-scripted.pt"
    else:
        raise HTTPException(422, "fmt must be one of: pt, json, scripted")

    if not path.exists():
        raise HTTPException(404, "model file missing on disk")
    return FileResponse(path, filename=path.name)
