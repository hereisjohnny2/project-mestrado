"""Login/logout (HttpOnly session cookie) and the user area. There is no
self-service sign-up: accounts are created by an admin with
``python -m app.create_user``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.security import create_access_token, hash_password, verify_password
from ..db import models
from ..db.session import get_db
from ..schemas import LoginRequest, MyModelOut, MyProjectOut, UserOut
from .deps import COOKIE_NAME, get_current_user

router = APIRouter()

# Hash checked when the email is unknown, so login takes the same time
# whether or not the account exists.
_DUMMY_HASH = hash_password("not-a-real-password")


def _set_session_cookie(response: Response, user: models.User) -> None:
    settings = get_settings()
    response.set_cookie(
        COOKIE_NAME,
        create_access_token(user.id),
        max_age=settings.token_ttl_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


@router.post("/auth/login", response_model=UserOut)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> models.User:
    user = db.query(models.User).filter(models.User.email == payload.email.strip().lower()).first()
    ok = verify_password(payload.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok:
        raise HTTPException(401, "invalid email or password")
    _set_session_cookie(response, user)
    return user


@router.post("/auth/logout", status_code=204, response_model=None)
def logout(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


@router.get("/auth/me", response_model=UserOut)
def me(user: models.User = Depends(get_current_user)) -> models.User:
    return user


@router.get("/me/projects", response_model=list[MyProjectOut])
def my_projects(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[MyProjectOut]:
    out: list[MyProjectOut] = []
    for project in (
        db.query(models.Project).filter(models.Project.owner_id == user.id).order_by(models.Project.created_at.desc())
    ):
        n_models = (
            db.query(func.count(models.MLModel.id))
            .join(models.Dataset)
            .filter(models.Dataset.project_id == project.id)
            .scalar()
        )
        out.append(
            MyProjectOut(
                id=project.id,
                name=project.name,
                classes=project.classes,
                created_at=project.created_at,
                n_images=len(project.images),
                n_annotated=sum(1 for img in project.images if img.annotation is not None),
                n_datasets=len(project.datasets),
                n_models=n_models,
            )
        )
    return out


@router.get("/me/models", response_model=list[MyModelOut])
def my_models(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[MyModelOut]:
    rows = (
        db.query(models.MLModel, models.Project)
        .join(models.Dataset, models.MLModel.dataset_id == models.Dataset.id)
        .join(models.Project, models.Dataset.project_id == models.Project.id)
        .filter(models.Project.owner_id == user.id)
        .order_by(models.MLModel.created_at.desc())
        .all()
    )
    return [
        MyModelOut(
            id=m.id,
            dataset_id=m.dataset_id,
            name=m.name,
            version=m.version,
            architecture=m.architecture,
            classes=m.classes,
            metrics=m.metrics,
            config=m.config,
            created_at=m.created_at,
            project_id=p.id,
            project_name=p.name,
        )
        for m, p in rows
    ]
