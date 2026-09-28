"""Fase 2 acceptance criterion (plan §6): generate a dataset from annotated
masks and train a model on it through the API, end to end."""

from __future__ import annotations

import io
import json
import time

from PIL import Image as PILImage


def _image_bytes(size=(6, 6), color=(120, 60, 30)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_bytes(size, pore_box, solid_box) -> bytes:
    """A grayscale-as-RGB mask (see AnnotationCanvas.tsx): 0 unlabeled,
    1 = poro inside pore_box, 2 = solido inside solid_box."""
    img = PILImage.new("RGB", size, (0, 0, 0))
    px = img.load()
    x0, y0, x1, y1 = pore_box
    for x in range(x0, x1):
        for y in range(y0, y1):
            px[x, y] = (1, 1, 1)
    x0, y0, x1, y1 = solid_box
    for x in range(x0, x1):
        for y in range(y0, y1):
            px[x, y] = (2, 2, 2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_annotated_project(api_client, size=(10, 10)):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("rock.png", _image_bytes(size), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]

    mask = _mask_bytes(size, pore_box=(0, 0, 4, 10), solid_box=(4, 0, 10, 10))
    resp = api_client.put(f"/images/{image['id']}/mask", content=mask, headers={"content-type": "image/png"})
    assert resp.status_code == 200
    return project, image


def test_generate_dataset_from_masks(api_client):
    project, _ = _make_annotated_project(api_client)

    resp = api_client.post(f"/projects/{project['id']}/datasets")
    assert resp.status_code == 201
    dataset = resp.json()
    assert dataset["n_pore"] == 40
    assert dataset["n_solid"] == 60
    assert dataset["n_pixels"] == 100

    resp = api_client.get(f"/datasets/{dataset['id']}/stats")
    assert resp.status_code == 200
    assert resp.json()["n_pore"] == 40

    resp = api_client.get(f"/datasets/{dataset['id']}/histogram")
    assert resp.status_code == 200
    body = resp.json()
    assert sum(body["pore"]["r"]) == 40
    assert sum(body["solid"]["r"]) == 60

    resp = api_client.get(f"/datasets/{dataset['id']}/download")
    assert resp.status_code == 200
    lines = resp.content.decode().strip().splitlines()
    assert len(lines) == 100


def test_dataset_generation_fails_without_annotations(api_client):
    project = api_client.post("/projects", json={"name": "empty"}).json()
    resp = api_client.post(f"/projects/{project['id']}/datasets")
    assert resp.status_code == 422


def test_training_job_and_publish_round_trip(api_client):
    project, _ = _make_annotated_project(api_client, size=(12, 12))
    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()

    resp = api_client.post(
        "/training/jobs",
        json={"dataset_id": dataset["id"], "epochs": 2, "batch_size": 8, "seed": 42},
    )
    assert resp.status_code == 201
    job = resp.json()
    assert job["status"] in ("pending", "running")

    deadline = time.time() + 30
    while time.time() < deadline:
        job = api_client.get(f"/training/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job.get("error")
    assert job["metrics"] is not None
    assert 0.0 <= job["metrics"]["accuracy"] <= 1.0

    resp = api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "modelo-1"})
    assert resp.status_code == 201
    model = resp.json()
    assert model["version"] == 1
    assert model["name"] == "modelo-1"

    # publishing again under the same name bumps the version
    resp = api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "modelo-1"})
    assert resp.json()["version"] == 2

    resp = api_client.get(f"/projects/{project['id']}/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = api_client.get(f"/models/{model['id']}/download?fmt=pt")
    assert resp.status_code == 200
    resp = api_client.get(f"/models/{model['id']}/download?fmt=json")
    assert resp.status_code == 200
    assert json.loads(resp.content)["format_version"] == 1
    resp = api_client.get(f"/models/{model['id']}/download?fmt=scripted")
    assert resp.status_code == 200


def test_training_job_stream_reports_progress(api_client):
    project, _ = _make_annotated_project(api_client, size=(12, 12))
    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()

    job = api_client.post(
        "/training/jobs",
        json={"dataset_id": dataset["id"], "epochs": 1, "batch_size": 8, "seed": 1},
    ).json()

    with api_client.stream("GET", f"/training/jobs/{job['id']}/stream") as resp:
        assert resp.status_code == 200
        final = None
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            final = json.loads(line[len("data: "):])
            if final["status"] in ("done", "failed"):
                break
        assert final is not None
        assert final["status"] == "done"
