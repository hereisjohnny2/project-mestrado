"""Fase 4: importing artifacts produced by the legacy CLI (plan §6)."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image as PILImage

from rock_model import RockNetModel as LegacyRockNetModel
from utils.image import apply_binarization as legacy_apply_binarization
from utils.image import calculate_porosity as legacy_calculate_porosity

from .test_segmentation import _noise_image_bytes, _wait

DAT = "\n".join(["200\t200\t200\tSolido"] * 30 + ["20\t20\t20\tPoro"] * 20) + "\n"


def _project(api_client):
    return api_client.post("/projects", json={"name": "legacy"}).json()


def _import_dat(api_client, project, content=DAT, filename="a.dat"):
    return api_client.post(
        f"/projects/{project['id']}/datasets/import",
        files={"file": (filename, content.encode(), "text/plain")},
    )


def test_import_dat_counts_and_downloads_identically(api_client):
    project = _project(api_client)
    resp = _import_dat(api_client, project)
    assert resp.status_code == 201
    ds = resp.json()
    assert (ds["n_pixels"], ds["n_pore"], ds["n_solid"]) == (50, 20, 30)

    downloaded = api_client.get(f"/datasets/{ds['id']}/download").content.decode()
    assert downloaded == DAT
    assert api_client.get(f"/datasets/{ds['id']}/stats").json()["n_pore"] == 20


def test_import_dat_rejects_malformed_lines(api_client):
    project = _project(api_client)
    for bad in ["1\t2\tPoro\n", "1\t2\t300\tPoro\n", "a\tb\tc\tPoro\n", "1\t2\t3\tPoro\n\n", ""]:
        resp = _import_dat(api_client, project, content=bad)
        assert resp.status_code == 422, bad
    assert api_client.get(f"/projects/{project['id']}/datasets").json() == []


def test_import_dat_reports_line_number(api_client):
    project = _project(api_client)
    resp = _import_dat(api_client, project, content="1\t2\t3\tPoro\nbroken\n")
    assert "linha 2" in resp.json()["detail"]


def _legacy_pt_bytes(seed=0) -> bytes:
    torch.manual_seed(seed)
    buf = io.BytesIO()
    torch.save(LegacyRockNetModel().state_dict(), buf)
    return buf.getvalue()


def _import_pt(api_client, project, dataset_id, content, name="antigo"):
    return api_client.post(
        f"/projects/{project['id']}/models/import",
        data={"name": name, "dataset_id": dataset_id},
        files={"file": ("dataset-nn-model.pt", content, "application/octet-stream")},
    )


def test_import_pt_evaluates_and_segments_like_legacy(api_client, tmp_path):
    project = _project(api_client)
    dataset = _import_dat(api_client, project).json()

    resp = _import_pt(api_client, project, dataset["id"], _legacy_pt_bytes(seed=5))
    assert resp.status_code == 201, resp.text
    model = resp.json()
    assert model["config"]["imported"] is True
    assert model["version"] == 1
    assert 0.0 <= model["metrics"]["accuracy"] <= 1.0
    (tn, fp), (fn, tp) = model["metrics"]["confusion_matrix"]
    assert tn + fp + fn + tp == 50
    assert model["metrics"]["loss_curve"] == []

    # second import under the same name bumps the version
    assert _import_pt(api_client, project, dataset["id"], _legacy_pt_bytes(1)).json()["version"] == 2

    # every download format exists, incl. the TorchScript export
    for fmt in ("pt", "json", "scripted"):
        assert api_client.get(f"/models/{model['id']}/download", params={"fmt": fmt}).status_code == 200

    # and the imported weights segment exactly like the legacy pipeline
    files = [("files", ("n.png", _noise_image_bytes(seed=9), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]
    run = api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).json()
    assert _wait(api_client, f"/segmentation/runs/{run['id']}")["status"] == "done"
    result = api_client.get(f"/segmentation/runs/{run['id']}/results").json()[0]

    torch.manual_seed(5)
    legacy_net = LegacyRockNetModel()
    legacy_net.eval()
    img = tmp_path / "n.png"
    img.write_bytes(_noise_image_bytes(seed=9))
    legacy_mask = legacy_apply_binarization(str(img), legacy_net)
    if isinstance(legacy_mask, tuple):
        legacy_mask = legacy_mask[0]
    assert abs(result["porosity"] - legacy_calculate_porosity(legacy_mask)) < 1e-9


def test_import_pt_rejects_incompatible_files(api_client):
    project = _project(api_client)
    dataset = _import_dat(api_client, project).json()

    assert _import_pt(api_client, project, dataset["id"], b"not a torch file").status_code == 422

    buf = io.BytesIO()
    torch.save({"weight": torch.zeros(2, 2)}, buf)
    assert _import_pt(api_client, project, dataset["id"], buf.getvalue()).status_code == 422

    scripted = io.BytesIO()
    torch.jit.save(torch.jit.script(LegacyRockNetModel()), scripted)
    assert _import_pt(api_client, project, dataset["id"], scripted.getvalue()).status_code == 422

    assert _import_pt(api_client, project, dataset["id"], _legacy_pt_bytes(), name="../x").status_code == 422
    assert _import_pt(api_client, project, "nope", _legacy_pt_bytes()).status_code == 404
    assert api_client.get(f"/projects/{project['id']}/models").json() == []


def test_seed_script_populates_a_project(api_client):
    from app import seed

    seed.seed(api_client)
    seed.seed(api_client)  # idempotent

    projects = api_client.get("/projects").json()
    assert len(projects) == 1
    models_ = api_client.get(f"/projects/{projects[0]['id']}/models").json()
    assert len(models_) == 1
    assert models_[0]["metrics"]["accuracy"] > 0.9
