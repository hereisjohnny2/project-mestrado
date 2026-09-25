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
