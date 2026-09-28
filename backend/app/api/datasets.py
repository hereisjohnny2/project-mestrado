"""Dataset generation and inspection (plan §4.4, Fase 2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.storage import ProjectStorage
from ..db import models
from ..db.session import get_db
from ..ml.dataset import dataset_histogram, dataset_stats
from ..schemas import DatasetHistogramOut, DatasetOut
from ..services.dataset import (
    EmptyDatasetError,
    InvalidDatasetError,
    dataset_absolute_path,
    generate_dataset,
    import_dataset,
)

router = APIRouter()


@router.post("/projects/{project_id}/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(project_id: str, db: Session = Depends(get_db)) -> models.Dataset:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")

    storage = ProjectStorage(project.id)
    try:
        fields = generate_dataset(project, storage)
    except EmptyDatasetError as exc:
        raise HTTPException(422, str(exc)) from exc

    dataset = models.Dataset(
        id=fields["id"],
        project_id=project.id,
        path=fields["path"],
        n_pixels=fields["n_pixels"],
        n_pore=fields["n_pore"],
        n_solid=fields["n_solid"],
        sha256=fields["sha256"],
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.post("/projects/{project_id}/datasets/import", response_model=DatasetOut, status_code=201)
def import_legacy_dataset(project_id: str, file: UploadFile, db: Session = Depends(get_db)) -> models.Dataset:
    """Imports a ``.dat`` produced by the legacy Qt annotation app."""
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")

    try:
        fields = import_dataset(file.file, ProjectStorage(project.id))
    except InvalidDatasetError as exc:
        raise HTTPException(422, str(exc)) from exc

    dataset = models.Dataset(project_id=project.id, **fields)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: str, db: Session = Depends(get_db)) -> list[models.Dataset]:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return db.query(models.Dataset).filter(models.Dataset.project_id == project_id).order_by(models.Dataset.created_at.desc()).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> models.Dataset:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/stats")
def get_dataset_stats(dataset_id: str, db: Session = Depends(get_db)) -> dict:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    return dataset_stats(dataset_absolute_path(dataset))


@router.get("/datasets/{dataset_id}/histogram", response_model=DatasetHistogramOut)
def get_dataset_histogram(dataset_id: str, db: Session = Depends(get_db)) -> dict:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    return dataset_histogram(dataset_absolute_path(dataset))


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, db: Session = Depends(get_db)) -> FileResponse:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    path = dataset_absolute_path(dataset)
    if not path.exists():
        raise HTTPException(404, "dataset file missing on disk")
    return FileResponse(path, media_type="text/tab-separated-values", filename=f"{dataset_id}.dat")
