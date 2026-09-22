"""SQLAlchemy models — mirrors plan §4.2.

Kept intentionally simple for Fase 0: enough structure for the API layer
built in later phases to attach to, without over-designing ahead of the
screens that will actually drive these tables.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class AnnotationClass(str, enum.Enum):
    PORE = "poro"
    SOLID = "solido"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    images: Mapped[list["Image"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped[Project] = relationship(back_populates="images")
    annotation: Mapped["Annotation | None"] = relationship(back_populates="image", uselist=False, cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), unique=True)
    mask_path: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    image: Mapped[Image] = relationship(back_populates="annotation")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    path: Mapped[str] = mapped_column(String, nullable=False)
    n_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    n_pore: Mapped[int] = mapped_column(Integer, nullable=False)
    n_solid: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped[Project] = relationship(back_populates="datasets")
    models: Mapped[list["MLModel"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    pt_path: Mapped[str] = mapped_column(String, nullable=False)
    json_path: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dataset: Mapped[Dataset] = relationship(back_populates="models")
    runs: Mapped[list["Run"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    model: Mapped[MLModel] = relationship(back_populates="runs")
    results: Mapped[list["Result"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"))
    porosity: Mapped[float] = mapped_column(Float, nullable=False)
    time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    bin_path: Mapped[str] = mapped_column(String, nullable=False)
    overlay_path: Mapped[str] = mapped_column(String, nullable=False)

    run: Mapped[Run] = relationship(back_populates="results")
