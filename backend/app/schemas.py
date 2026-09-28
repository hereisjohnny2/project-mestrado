"""Pydantic response/request models for the API layer (Fase 1: projects +
images + annotation mask)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str


class ProjectUpdate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageOut(BaseModel):
    id: str
    project_id: str
    filename: str
    width: int
    height: int
    sha256: str
    created_at: datetime
    has_annotation: bool

    model_config = {"from_attributes": True}


class ProjectDetailOut(ProjectOut):
    images: list[ImageOut]


class DatasetOut(BaseModel):
    id: str
    project_id: str
    n_pixels: int
    n_pore: int
    n_solid: int
    sha256: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetHistogramOut(BaseModel):
    bin_edges: list[float]
    pore: dict[str, list[int]]
    solid: dict[str, list[int]]


class TrainingJobCreate(BaseModel):
    dataset_id: str
    epochs: int = 5
    learning_rate: float = 0.0025
    batch_size: int = 16
    split_ratio: float = 0.8
    seed: int | None = None


class TrainingMetricsOut(BaseModel):
    accuracy: float
    loss_curve: list[float]
    confusion_matrix: list[list[int]]
    per_class: dict
    duration_ms: float


class TrainingJobOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    dataset_id: str
    status: str
    epoch: int
    epochs: int
    loss_curve: list[float]
    error: str | None
    model_id: str | None
    metrics: TrainingMetricsOut | None


class PublishModelRequest(BaseModel):
    name: str


class ModelOut(BaseModel):
    id: str
    dataset_id: str
    name: str
    version: int
    metrics: dict
    config: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentationRunCreate(BaseModel):
    model_id: str
    image_ids: list[str]


class SegmentationRunOut(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}

    id: str
    model_id: str
    status: str
    total: int
    done: int
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class SegmentationResultOut(BaseModel):
    id: str
    run_id: str
    image_id: str
    filename: str
    porosity: float
    time_ms: float
