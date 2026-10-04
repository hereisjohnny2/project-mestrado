"""In-process training job registry.

The app is single-process (plan §9: `docker compose up` local); jobs are
owned by the user who started them. A training run is CPU-bound and takes seconds to a few
minutes for the dataset sizes in scope, so a plain background thread plus an
in-memory job table is enough — no task queue, no persistence across
restarts. Progress is polled by the SSE endpoint every ``POLL_INTERVAL``.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from ..ml.training import TrainingConfig, TrainingResult, train

POLL_INTERVAL = 0.25


@dataclass
class JobState:
    id: str
    owner_id: str
    dataset_id: str
    dataset_path: str
    config: TrainingConfig
    class_names: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed
    epoch: int = 0
    epochs: int = 0
    loss_curve: list[float] = field(default_factory=list)
    result: TrainingResult | None = None
    error: str | None = None
    model_id: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "dataset_id": self.dataset_id,
                "architecture": self.config.architecture,
                "status": self.status,
                "epoch": self.epoch,
                "epochs": self.epochs,
                "loss_curve": list(self.loss_curve),
                "error": self.error,
                "model_id": self.model_id,
                "metrics": _metrics(self.result),
            }


def _metrics(result: TrainingResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "accuracy": result.accuracy,
        "loss_curve": result.loss_curve,
        "confusion_matrix": result.confusion_matrix,
        "per_class": result.per_class,
        "duration_ms": result.duration_ms,
    }


JOBS: dict[str, JobState] = {}


def start_job(owner_id: str, dataset_id: str, dataset_path: str, config: TrainingConfig, class_names: list[str]) -> JobState:
    job = JobState(
        id=str(uuid.uuid4()),
        owner_id=owner_id,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        config=config,
        class_names=list(class_names),
        epochs=config.epochs,
    )
    JOBS[job.id] = job

    def on_epoch_end(epoch_index: int, epochs: int, loss_value: float | None) -> None:
        with job._lock:
            job.epoch = epoch_index
            job.epochs = epochs
            if loss_value is not None:
                job.loss_curve.append(loss_value)

    def run() -> None:
        with job._lock:
            job.status = "running"
        try:
            result = train(dataset_path, config, on_epoch_end=on_epoch_end, class_names=job.class_names)
            with job._lock:
                job.result = result
                job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as job.error
            with job._lock:
                job.error = str(exc)
                job.status = "failed"

    threading.Thread(target=run, daemon=True).start()
    return job


def get_job(job_id: str) -> JobState | None:
    return JOBS.get(job_id)


def wait_for_update(job: JobState, last_epoch: int, last_status: str, timeout: float = 5.0) -> None:
    """Blocks (via polling) until the job's epoch/status changes or `timeout`
    elapses — keeps the SSE generator from busy-spinning between events."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with job._lock:
            if job.epoch != last_epoch or job.status != last_status:
                return
        time.sleep(POLL_INTERVAL)
