"""Batch segmentation runs, results and export (plan §4.4, Fase 3)."""

from __future__ import annotations

import csv
import io
import time
import zipfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..db import models
from ..db.session import get_db
from ..schemas import SegmentationResultOut, SegmentationRunCreate, SegmentationRunOut
from ..services.segmentation_runs import run_progress, start_run

router = APIRouter()

POLL_INTERVAL = 0.25


def _run_out(run: models.Run) -> SegmentationRunOut:
    done, total = run_progress(run)
    return SegmentationRunOut(
        id=run.id,
        model_id=run.model_id,
        status=run.status.value,
        total=total,
        done=done,
        error=run.error,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )


def _result_out(result: models.Result) -> SegmentationResultOut:
    return SegmentationResultOut(
        id=result.id,
        run_id=result.run_id,
        image_id=result.image_id,
        filename=result.image.filename,
        porosity=result.porosity,
        time_ms=result.time_ms,
    )


def _get_run(db: Session, run_id: str) -> models.Run:
    run = db.get(models.Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.post("/segmentation/runs", response_model=SegmentationRunOut, status_code=201)
def create_run(payload: SegmentationRunCreate, db: Session = Depends(get_db)) -> SegmentationRunOut:
    model = db.get(models.MLModel, payload.model_id)
    if model is None:
        raise HTTPException(404, "model not found")
    if not payload.image_ids:
        raise HTTPException(422, "image_ids must not be empty")

    image_ids = list(dict.fromkeys(payload.image_ids))
    for image_id in image_ids:
        image = db.get(models.Image, image_id)
        if image is None:
            raise HTTPException(404, f"image {image_id} not found")
        if image.project_id != model.dataset.project_id:
            raise HTTPException(422, f"image {image_id} belongs to a different project than the model")

    run = models.Run(model_id=model.id)
    db.add(run)
    db.commit()
    db.refresh(run)
    start_run(run.id, image_ids)
    return _run_out(run)


@router.get("/projects/{project_id}/runs", response_model=list[SegmentationRunOut])
def list_runs(project_id: str, db: Session = Depends(get_db)) -> list[SegmentationRunOut]:
    if db.get(models.Project, project_id) is None:
        raise HTTPException(404, "project not found")
    runs = (
        db.query(models.Run)
        .join(models.MLModel)
        .join(models.Dataset)
        .filter(models.Dataset.project_id == project_id)
        .order_by(models.Run.created_at.desc())
        .all()
    )
    return [_run_out(run) for run in runs]


@router.get("/segmentation/runs/{run_id}", response_model=SegmentationRunOut)
def get_run(run_id: str, db: Session = Depends(get_db)) -> SegmentationRunOut:
    return _run_out(_get_run(db, run_id))


@router.delete("/segmentation/runs/{run_id}", status_code=204, response_model=None)
def delete_run(run_id: str, db: Session = Depends(get_db)) -> None:
    import shutil

    run = _get_run(db, run_id)
    if run.status in (models.RunStatus.PENDING, models.RunStatus.RUNNING):
        raise HTTPException(409, "run is still in progress")
    project_id = run.model.dataset.project_id
    db.delete(run)
    db.commit()
    shutil.rmtree(get_settings().storage_dir / project_id / "runs" / run_id, ignore_errors=True)


@router.get("/segmentation/runs/{run_id}/stream")
def stream_run(run_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    _get_run(db, run_id)

    def events():
        from ..db.session import get_session_factory

        last = None
        while True:
            with get_session_factory()() as session:
                out = _run_out(_get_run(session, run_id))
            state = (out.done, out.status)
            if state != last:
                last = state
                yield f"data: {out.model_dump_json()}\n\n"
            if out.status in ("done", "failed"):
                return
            time.sleep(POLL_INTERVAL)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/segmentation/runs/{run_id}/results", response_model=list[SegmentationResultOut])
def run_results(run_id: str, db: Session = Depends(get_db)) -> list[SegmentationResultOut]:
    run = _get_run(db, run_id)
    return [_result_out(r) for r in run.results]


def _result_file(db: Session, result_id: str, attr: str) -> FileResponse:
    result = db.get(models.Result, result_id)
    if result is None:
        raise HTTPException(404, "result not found")
    path = get_settings().storage_dir / getattr(result, attr)
    if not path.exists():
        raise HTTPException(404, "result file missing on disk")
    return FileResponse(path, media_type="image/png")


@router.get("/segmentation/results/{result_id}/bin")
def result_bin(result_id: str, db: Session = Depends(get_db)) -> FileResponse:
    return _result_file(db, result_id, "bin_path")


@router.get("/segmentation/results/{result_id}/overlay")
def result_overlay(result_id: str, db: Session = Depends(get_db)) -> FileResponse:
    return _result_file(db, result_id, "overlay_path")


@router.get("/segmentation/runs/{run_id}/export")
def export_run(run_id: str, fmt: str = "csv", db: Session = Depends(get_db)) -> Response:
    if fmt not in ("csv", "zip"):
        raise HTTPException(422, "fmt must be one of: csv, zip")
    run = _get_run(db, run_id)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["nome", "porosidade", "tempo_ms"])
    for r in run.results:
        writer.writerow([r.image.filename, r.porosity, r.time_ms])
    csv_text = buf.getvalue()

    if fmt == "csv":
        return Response(
            csv_text,
            media_type="text/csv",
            headers={"content-disposition": f'attachment; filename="porosity-{run.id[:8]}.csv"'},
        )

    storage_dir = get_settings().storage_dir
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("porosity.csv", csv_text)
        for r in run.results:
            stem = r.image.filename.rsplit(".", 1)[0]
            zf.write(storage_dir / r.bin_path, f"{stem}-bin.png")
    return Response(
        zbuf.getvalue(),
        media_type="application/zip",
        headers={"content-disposition": f'attachment; filename="segmentation-{run.id[:8]}.zip"'},
    )
