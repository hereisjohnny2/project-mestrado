"""Per-project classes: editing them, generating a multi-class dataset,
and training/segmenting with the binary (1.0.0) architecture on top."""

from __future__ import annotations

import csv
import io

import numpy as np
from PIL import Image as PILImage

from .test_dataset_and_training import _image_bytes
from .test_segmentation import _wait

THREE = [
    {"index": 1, "name": "Poro", "color": "#2563eb"},
    {"index": 2, "name": "Sólido", "color": "#ea580c"},
    {"index": 3, "name": "Cimento", "color": "#16a34a"},
]


def _index_mask_bytes(size, boxes: dict[int, tuple[int, int, int, int]]) -> bytes:
    img = PILImage.new("L", size, 0)
    px = img.load()
    for value, (x0, y0, x1, y1) in boxes.items():
        for x in range(x0, x1):
            for y in range(y0, y1):
                px[x, y] = value
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _three_class_project(api_client, size=(12, 12)):
    project = api_client.post("/projects", json={"name": "p"}).json()
    assert api_client.put(f"/projects/{project['id']}/classes", json={"classes": THREE}).status_code == 200
    files = [("files", ("rock.png", _image_bytes(size), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]
    mask = _index_mask_bytes(size, {1: (0, 0, 4, 12), 2: (4, 0, 8, 12), 3: (8, 0, 12, 12)})
    assert api_client.put(f"/images/{image['id']}/mask", content=mask, headers={"content-type": "image/png"}).status_code == 200
    return project, image


def test_new_project_has_default_classes(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    assert [c["name"] for c in project["classes"]] == ["Poro", "Sólido"]
    assert api_client.get(f"/projects/{project['id']}/classes").json() == project["classes"]
    assert api_client.get(f"/projects/{project['id']}").json()["classes"] == project["classes"]


def test_classes_validation(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    url = f"/projects/{project['id']}/classes"

    def put(classes):
        return api_client.put(url, json={"classes": classes})

    assert put([]).status_code == 422
    assert put([{"index": 2, "name": "A", "color": "#000000"}]).status_code == 422
    assert put([{"index": 1, "name": " ", "color": "#000000"}]).status_code == 422
    assert put([{"index": 1, "name": "A", "color": "red"}]).status_code == 422
    assert put([{"index": 1, "name": "A", "color": "#000000"}, {"index": 2, "name": "a", "color": "#111111"}]).status_code == 422

    resp = put([{"index": 1, "name": " Poros ", "color": "#ABCDEF"}])
    assert resp.status_code == 200
    assert resp.json() == [{"index": 1, "name": "Poros", "color": "#abcdef"}]


def test_class_removal_blocked_while_masks_use_it(api_client):
    project, image = _three_class_project(api_client)
    url = f"/projects/{project['id']}/classes"

    resp = api_client.put(url, json={"classes": THREE[:2]})
    assert resp.status_code == 409
    assert "Cimento" in resp.json()["detail"]

    # renaming is always fine
    renamed = [dict(THREE[0]), dict(THREE[1]), {**THREE[2], "name": "Matriz"}]
    assert api_client.put(url, json={"classes": renamed}).status_code == 200

    api_client.delete(f"/images/{image['id']}/mask")
    assert api_client.put(url, json={"classes": THREE[:2]}).status_code == 200


def test_multiclass_dataset_and_binary_training(api_client, tmp_path):
    project, image = _three_class_project(api_client)

    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()
    assert dataset["class_counts"] == {"Poro": 48, "Sólido": 48, "Cimento": 48}
    assert [c["name"] for c in dataset["classes"]] == ["Poro", "Sólido", "Cimento"]
    labels = {line.split("\t")[3] for line in api_client.get(f"/datasets/{dataset['id']}/download").text.splitlines()}
    assert labels == {"Poro", "Sólido", "Cimento"}

    hist = api_client.get(f"/datasets/{dataset['id']}/histogram").json()
    assert set(hist["classes"]) == {"Poro", "Sólido", "Cimento"}
    assert sum(hist["classes"]["Cimento"]["g"]) == 48

    # The legacy architecture is "pore vs. rest": two outputs, named accordingly.
    job = api_client.post("/training/jobs", json={"dataset_id": dataset["id"], "epochs": 1, "batch_size": 16, "seed": 3}).json()
    assert _wait(api_client, f"/training/jobs/{job['id']}")["status"] == "done"
    model = api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "bin"}).json()
    assert model["config"]["class_names"] == ["Outros", "Poro"]
    assert set(model["metrics"]["per_class"]) == {"Outros", "Poro"}
    assert len(model["metrics"]["confusion_matrix"]) == 2
    assert [c["name"] for c in model["classes"]] == ["Poro", "Sólido", "Cimento"]

    run = api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).json()
    assert _wait(api_client, f"/segmentation/runs/{run['id']}")["status"] == "done"
    result = api_client.get(f"/segmentation/runs/{run['id']}/results").json()[0]
    assert set(result["class_fractions"]) == {"Outros", "Poro"}
    assert result["porosity"] == result["class_fractions"]["Poro"]

    bin_img = PILImage.open(io.BytesIO(api_client.get(f"/segmentation/results/{result['id']}/bin").content))
    assert set(np.unique(np.asarray(bin_img))) <= {0, 1}

    rows = list(csv.reader(io.StringIO(api_client.get(f"/segmentation/runs/{run['id']}/export").text)))
    assert rows[0] == ["nome", "porosidade", "tempo_ms", "Outros", "Poro"]


def test_init_db_upgrades_old_schema(tmp_path, monkeypatch):
    """A database from before class lists existed gains the new columns and
    its rows get the equivalent Poro/Sólido values."""
    import json
    import sqlite3

    from app.core import config as config_module
    from app.db import session as session_module

    db_path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (id VARCHAR PRIMARY KEY, email VARCHAR, name VARCHAR, password_hash VARCHAR, created_at DATETIME);
        CREATE TABLE projects (id VARCHAR PRIMARY KEY, owner_id VARCHAR, name VARCHAR NOT NULL, created_at DATETIME);
        CREATE TABLE images (id VARCHAR PRIMARY KEY, project_id VARCHAR, filename VARCHAR, path VARCHAR, width INTEGER, height INTEGER, sha256 VARCHAR, created_at DATETIME);
        CREATE TABLE annotations (id VARCHAR PRIMARY KEY, image_id VARCHAR, mask_path VARCHAR, updated_at DATETIME);
        CREATE TABLE datasets (id VARCHAR PRIMARY KEY, project_id VARCHAR, path VARCHAR NOT NULL, n_pixels INTEGER NOT NULL,
            n_pore INTEGER NOT NULL, n_solid INTEGER NOT NULL, sha256 VARCHAR NOT NULL, created_at DATETIME);
        CREATE TABLE models (id VARCHAR PRIMARY KEY, dataset_id VARCHAR, name VARCHAR NOT NULL, version INTEGER NOT NULL,
            pt_path VARCHAR NOT NULL, json_path VARCHAR NOT NULL, metrics JSON NOT NULL, config JSON NOT NULL, created_at DATETIME);
        CREATE TABLE runs (id VARCHAR PRIMARY KEY, model_id VARCHAR, status VARCHAR, created_at DATETIME, finished_at DATETIME, error VARCHAR);
        CREATE TABLE results (id VARCHAR PRIMARY KEY, run_id VARCHAR, image_id VARCHAR, porosity FLOAT NOT NULL, time_ms FLOAT NOT NULL,
            bin_path VARCHAR NOT NULL, overlay_path VARCHAR NOT NULL);
        INSERT INTO projects VALUES ('p1', NULL, 'old', NULL);
        INSERT INTO datasets VALUES ('d1', 'p1', 'x.dat', 100, 40, 60, 'abc', NULL);
        INSERT INTO models VALUES ('m1', 'd1', 'm', 1, 'a.pt', 'a.json', '{}', '{}', NULL);
        INSERT INTO results VALUES ('r1', 'run1', 'i1', 0.25, 10.0, 'b.png', 'o.png');
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("ROCKSEG_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ROCKSEG_DATABASE_URL", f"sqlite:///{db_path}")
    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None
    try:
        session_module.init_db()
    finally:
        config_module.get_settings.cache_clear()
        session_module._engine = None
        session_module._SessionLocal = None

    conn = sqlite3.connect(db_path)
    project_classes = json.loads(conn.execute("SELECT classes FROM projects WHERE id='p1'").fetchone()[0])
    assert [c["name"] for c in project_classes] == ["Poro", "Sólido"]
    ds = conn.execute("SELECT classes, class_counts FROM datasets WHERE id='d1'").fetchone()
    assert json.loads(ds[1]) == {"Poro": 40, "Sólido": 60}
    assert {c[1] for c in conn.execute("PRAGMA table_info(datasets)")} >= {"classes", "class_counts"}
    assert "n_pore" not in {c[1] for c in conn.execute("PRAGMA table_info(datasets)")}
    assert conn.execute("SELECT architecture FROM models WHERE id='m1'").fetchone()[0] == "1.0.0"
    assert json.loads(conn.execute("SELECT class_fractions FROM results WHERE id='r1'").fetchone()[0]) == {"Poro": 0.25, "Sólido": 0.75}
    conn.close()
