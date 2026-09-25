"""Projects, image upload/listing, and mask persistence (plan §4.4, Fase 1)."""

from __future__ import annotations

import shutil
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.storage import ProjectStorage, get_settings
from ..db import models
from ..db.session import get_db
from ..schemas import ImageOut, ProjectCreate, ProjectDetailOut, ProjectOut, ProjectUpdate
from ..services.images import UnsupportedImageError, ingest_image, mask_relative_path

router = APIRouter()


def _image_out(image: models.Image) -> ImageOut:
    return ImageOut(
        id=image.id,
        project_id=image.project_id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        sha256=image.sha256,
        created_at=image.created_at,
        has_annotation=image.annotation is not None,
    )


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> models.Project:
    project = models.Project(name=payload.name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)) -> list[models.Project]:
    return db.query(models.Project).order_by(models.Project.created_at.desc()).all()


@router.get("/projects/{project_id}", response_model=ProjectDetailOut)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectDetailOut:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return ProjectDetailOut(
        id=project.id,
        name=project.name,
        created_at=project.created_at,
        images=[_image_out(img) for img in project.images],
    )


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def rename_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)) -> models.Project:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    project.name = payload.name
    db.commit()
    db.refresh(project)
    return project


@router.delete("/projects/{project_id}", status_code=204, response_model=None)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    db.delete(project)
    db.commit()
    shutil.rmtree(get_settings().storage_dir / project_id, ignore_errors=True)


@router.post("/projects/{project_id}/images", response_model=list[ImageOut], status_code=201)
async def upload_images(
    project_id: str, files: list[UploadFile], db: Session = Depends(get_db)
) -> list[ImageOut]:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")

    storage = ProjectStorage(project.id)
    created: list[models.Image] = []
    for file in files:
        raw = await file.read()
        try:
            fields = ingest_image(storage, file.filename or "upload", raw)
        except UnsupportedImageError as exc:
            raise HTTPException(422, str(exc)) from exc
        image = models.Image(
            id=fields["id"],
            project_id=project.id,
            filename=fields["filename"],
            path=fields["path"],
            width=fields["width"],
            height=fields["height"],
            sha256=fields["sha256"],
        )
        db.add(image)
        created.append(image)

    db.commit()
    for image in created:
        db.refresh(image)
    return [_image_out(image) for image in created]


@router.get("/images/{image_id}", response_model=ImageOut)
def get_image(image_id: str, db: Session = Depends(get_db)) -> ImageOut:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return _image_out(image)


@router.delete("/images/{image_id}", status_code=204, response_model=None)
def delete_image(image_id: str, db: Session = Depends(get_db)) -> None:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    storage_dir = get_settings().storage_dir
    (storage_dir / image.path).unlink(missing_ok=True)
    if image.annotation is not None:
        (storage_dir / image.annotation.mask_path).unlink(missing_ok=True)
    db.delete(image)
    db.commit()


@router.get("/images/{image_id}/file")
def get_image_file(image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    path = get_settings().storage_dir / image.path
    if not path.exists():
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(path, media_type="image/png")


@router.get("/images/{image_id}/mask")
def get_image_mask(image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    if image.annotation is None:
        raise HTTPException(404, "no annotation yet")
    path = get_settings().storage_dir / image.annotation.mask_path
    if not path.exists():
        raise HTTPException(404, "mask file missing on disk")
    return FileResponse(path, media_type="image/png")


@router.put("/images/{image_id}/mask", response_model=ImageOut)
async def put_image_mask(image_id: str, request: Request, db: Session = Depends(get_db)) -> ImageOut:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")

    raw = await request.body()
    if not raw:
        raise HTTPException(422, "empty mask body")

    storage = ProjectStorage(image.project_id)
    dest, rel_path = mask_relative_path(storage, image.id)
    dest.write_bytes(raw)

    if image.annotation is None:
        image.annotation = models.Annotation(image_id=image.id, mask_path=rel_path)
        db.add(image.annotation)
    else:
        image.annotation.mask_path = rel_path
        image.annotation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(image)
    return _image_out(image)


@router.delete("/images/{image_id}/mask", response_model=ImageOut)
def clear_image_mask(image_id: str, db: Session = Depends(get_db)) -> ImageOut:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    if image.annotation is not None:
        path = get_settings().storage_dir / image.annotation.mask_path
        path.unlink(missing_ok=True)
        db.delete(image.annotation)
        db.commit()
        db.refresh(image)
    return _image_out(image)
