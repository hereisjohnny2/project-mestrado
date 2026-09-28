"""Authentication dependency and owner-scoped lookups.

Every resource lookup goes through these helpers so a user can only reach
their own data. A resource owned by someone else is reported as 404, not
403, so ids can't be probed for existence.
"""

from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.security import decode_access_token
from ..db import models
from ..db.session import get_db

COOKIE_NAME = "rockseg_session"


def get_current_user(
    token: str | None = Cookie(default=None, alias=COOKIE_NAME), db: Session = Depends(get_db)
) -> models.User:
    user_id = decode_access_token(token) if token else None
    user = db.get(models.User, user_id) if user_id else None
    if user is None:
        raise HTTPException(401, "not authenticated")
    return user


def get_owned_project(db: Session, user: models.User, project_id: str) -> models.Project:
    project = db.get(models.Project, project_id)
    if project is None or project.owner_id != user.id:
        raise HTTPException(404, "project not found")
    return project


def get_owned_image(db: Session, user: models.User, image_id: str) -> models.Image:
    image = db.get(models.Image, image_id)
    if image is None or image.project.owner_id != user.id:
        raise HTTPException(404, "image not found")
    return image


def get_owned_dataset(db: Session, user: models.User, dataset_id: str) -> models.Dataset:
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None or dataset.project.owner_id != user.id:
        raise HTTPException(404, "dataset not found")
    return dataset


def get_owned_model(db: Session, user: models.User, model_id: str) -> models.MLModel:
    model = db.get(models.MLModel, model_id)
    if model is None or model.dataset.project.owner_id != user.id:
        raise HTTPException(404, "model not found")
    return model


def get_owned_run(db: Session, user: models.User, run_id: str) -> models.Run:
    run = db.get(models.Run, run_id)
    if run is None or run.model.dataset.project.owner_id != user.id:
        raise HTTPException(404, "run not found")
    return run


def get_owned_result(db: Session, user: models.User, result_id: str) -> models.Result:
    result = db.get(models.Result, result_id)
    if result is None or result.run.model.dataset.project.owner_id != user.id:
        raise HTTPException(404, "result not found")
    return result
