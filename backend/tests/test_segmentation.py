"""Fase 3 acceptance criterion (plan §6): the porosity computed by a web
segmentation run matches the legacy ``tester.py`` pipeline for the same
image and the same model."""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile

import numpy as np
import torch
from PIL import Image as PILImage

from rock_model import RockNetModel as LegacyRockNetModel
from utils.image import apply_binarization as legacy_apply_binarization
from utils.image import calculate_porosity as legacy_calculate_porosity

from .test_dataset_and_training import _make_annotated_project


def _wait(api_client, url, key="status", done=("done", "failed")):
    deadline = time.time() + 30
    while time.time() < deadline:
        body = api_client.get(url).json()
        if body[key] in done:
            return body
        time.sleep(0.1)
    raise AssertionError("timed out")


def _publish_model(api_client, project):
    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()
    job = api_client.post("/training/jobs", json={"dataset_id": dataset["id"], "epochs": 2, "batch_size": 8, "seed": 1}).json()
    assert _wait(api_client, f"/training/jobs/{job['id']}")["status"] == "done"
    return api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "m"}).json()


def _noise_image_bytes(size=(20, 15), seed=3) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_segmentation_run_matches_legacy_porosity(api_client, tmp_path):
    project, _ = _make_annotated_project(api_client, size=(12, 12))
    model = _publish_model(api_client, project)

    files = [("files", (f"n{i}.png", _noise_image_bytes(seed=i), "image/png")) for i in range(3)]
    images = api_client.post(f"/projects/{project['id']}/images", files=files).json()

    resp = api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [i["id"] for i in images]})
    assert resp.status_code == 201
    run = _wait(api_client, f"/segmentation/runs/{resp.json()['id']}")
    assert run["status"] == "done", run["error"]
    assert run["done"] == run["total"] == 3

    results = api_client.get(f"/segmentation/runs/{run['id']}/results").json()
    assert len(results) == 3

    # Legacy pipeline on the same weights and the same (RGB PNG) files.
    pt = api_client.get(f"/models/{model['id']}/download?fmt=pt").content
    pt_path = tmp_path / "m.pt"
    pt_path.write_bytes(pt)
    legacy = LegacyRockNetModel()
    legacy.load_state_dict(torch.load(pt_path))
    legacy.eval()

    for result, image in zip(sorted(results, key=lambda r: r["filename"]), sorted(images, key=lambda i: i["filename"])):
        img_path = tmp_path / image["filename"]
        img_path.write_bytes(api_client.get(f"/images/{image['id']}/file").content)
        mask, _ = legacy_apply_binarization(str(img_path), legacy)
        assert result["porosity"] == legacy_calculate_porosity(mask)
        assert result["class_fractions"]["Poro"] == result["porosity"]
        assert abs(sum(result["class_fractions"].values()) - 1.0) < 1e-9

        # Paletted PNG: pixel values are the class indices, rendered black/white.
        bin_img = PILImage.open(io.BytesIO(api_client.get(f"/segmentation/results/{result['id']}/bin").content))
        assert bin_img.mode == "P"
        assert np.array_equal(np.asarray(bin_img), mask)
        assert np.array_equal(np.asarray(bin_img.convert("L")) // 255, mask)
        assert api_client.get(f"/segmentation/results/{result['id']}/overlay").status_code == 200

    csv_rows = list(csv.reader(io.StringIO(api_client.get(f"/segmentation/runs/{run['id']}/export").text)))
    assert csv_rows[0] == ["nome", "porosidade", "tempo_ms", "Sólido", "Poro"]
    assert len(csv_rows) == 4

    z = zipfile.ZipFile(io.BytesIO(api_client.get(f"/segmentation/runs/{run['id']}/export?fmt=zip").content))
    assert sorted(z.namelist()) == ["n0-bin.png", "n1-bin.png", "n2-bin.png", "porosity.csv"]

    listed = api_client.get(f"/projects/{project['id']}/runs").json()
    assert [r["id"] for r in listed] == [run["id"]]


def test_run_stream_and_validation(api_client):
    project, image = _make_annotated_project(api_client, size=(12, 12))
    model = _publish_model(api_client, project)

    assert api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": []}).status_code == 422
    assert api_client.post("/segmentation/runs", json={"model_id": "nope", "image_ids": [image["id"]]}).status_code == 404
    other = api_client.post("/projects", json={"name": "o"}).json()
    foreign = api_client.post(
        f"/projects/{other['id']}/images", files=[("files", ("x.png", _noise_image_bytes(), "image/png"))]
    ).json()[0]
    assert api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [foreign["id"]]}).status_code == 422

    run = api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).json()
    with api_client.stream("GET", f"/segmentation/runs/{run['id']}/stream") as resp:
        events = [json.loads(l[6:]) for l in resp.iter_lines() if l.startswith("data: ")]
    assert events[-1]["status"] == "done"
    assert events[-1]["done"] == 1

    assert api_client.delete(f"/segmentation/runs/{run['id']}").status_code == 204
    assert api_client.get(f"/segmentation/runs/{run['id']}").status_code == 404
