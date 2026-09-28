"""Auth flow, ownership isolation between users, and the user area."""

from __future__ import annotations

import io

from PIL import Image

from .conftest import register


def _png(size=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_login_logout_me(anon_client):
    assert anon_client.get("/auth/me").status_code == 401

    user = register(anon_client, email="  Ana@Example.com ")
    assert user["email"] == "ana@example.com"
    assert "password" not in user and "password_hash" not in user
    assert anon_client.get("/auth/me").json()["id"] == user["id"]

    assert anon_client.post("/auth/logout").status_code == 204
    assert anon_client.get("/auth/me").status_code == 401

    resp = anon_client.post("/auth/login", json={"email": "ANA@example.com", "password": "senha-forte-123"})
    assert resp.status_code == 200
    assert anon_client.get("/auth/me").status_code == 200


def test_no_public_signup(anon_client):
    resp = anon_client.post("/auth/register", json={"email": "x@example.com", "name": "X", "password": "senha-forte-123"})
    assert resp.status_code in (404, 405)
    assert anon_client.get("/auth/me").status_code == 401


def test_login_rejects_bad_credentials(anon_client):
    register(anon_client)
    anon_client.post("/auth/logout")
    wrong = anon_client.post("/auth/login", json={"email": "ana@example.com", "password": "errada-errada"})
    unknown = anon_client.post("/auth/login", json={"email": "who@example.com", "password": "senha-forte-123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_tampered_or_garbage_cookie_is_rejected(anon_client):
    anon_client.cookies.set("rockseg_session", "not.a.jwt")
    assert anon_client.get("/projects").status_code == 401


def test_protected_routes_require_auth(anon_client):
    for method, path in [
        ("get", "/projects"),
        ("post", "/projects"),
        ("get", "/projects/x"),
        ("get", "/images/x/file"),
        ("get", "/datasets/x"),
        ("get", "/models/x/download"),
        ("get", "/training/jobs/x"),
        ("get", "/me/projects"),
        ("get", "/me/models"),
    ]:
        assert getattr(anon_client, method)(path).status_code == 401, (method, path)
    assert anon_client.get("/health").status_code == 200


def test_users_cannot_see_or_touch_each_others_data(app_factory):
    ana, bia = app_factory(), app_factory()
    register(ana)
    register(bia, email="bia@example.com", name="Bia")

    project = ana.post("/projects", json={"name": "ana-proj"}).json()
    image = ana.post(f"/projects/{project['id']}/images", files=[("files", ("a.png", _png(), "image/png"))]).json()[0]

    assert bia.get("/projects").json() == []
    assert bia.get("/me/projects").json() == []
    for resp in [
        bia.get(f"/projects/{project['id']}"),
        bia.patch(f"/projects/{project['id']}", json={"name": "hax"}),
        bia.delete(f"/projects/{project['id']}"),
        bia.post(f"/projects/{project['id']}/images", files=[("files", ("a.png", _png(), "image/png"))]),
        bia.post(f"/projects/{project['id']}/datasets"),
        bia.get(f"/projects/{project['id']}/models"),
        bia.get(f"/images/{image['id']}"),
        bia.get(f"/images/{image['id']}/file"),
        bia.put(f"/images/{image['id']}/mask", content=_png(), headers={"content-type": "image/png"}),
        bia.delete(f"/images/{image['id']}"),
    ]:
        assert resp.status_code == 404, resp.request.url

    # untouched for the owner
    assert ana.get(f"/projects/{project['id']}").json()["name"] == "ana-proj"


def test_user_area_lists_projects_with_counts_and_models(api_client):
    from .test_dataset_and_training import _make_annotated_project

    project, _ = _make_annotated_project(api_client)
    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()
    job = api_client.post("/training/jobs", json={"dataset_id": dataset["id"], "epochs": 1, "seed": 1}).json()
    # wait for the background job to finish
    import time

    for _ in range(200):
        if api_client.get(f"/training/jobs/{job['id']}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    model = api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "m"}).json()

    [row] = api_client.get("/me/projects").json()
    assert row["id"] == project["id"]
    assert (row["n_images"], row["n_annotated"], row["n_datasets"], row["n_models"]) == (1, 1, 1, 1)

    [mrow] = api_client.get("/me/models").json()
    assert mrow["id"] == model["id"]
    assert mrow["project_id"] == project["id"] and mrow["project_name"] == project["name"]


def test_training_job_and_model_are_private(app_factory):
    from .test_dataset_and_training import _make_annotated_project
    import time

    ana, bia = app_factory(), app_factory()
    register(ana)
    register(bia, email="bia@example.com", name="Bia")

    project, _ = _make_annotated_project(ana)
    dataset = ana.post(f"/projects/{project['id']}/datasets").json()
    job = ana.post("/training/jobs", json={"dataset_id": dataset["id"], "epochs": 1, "seed": 1}).json()

    assert bia.post("/training/jobs", json={"dataset_id": dataset["id"]}).status_code == 404
    assert bia.get(f"/training/jobs/{job['id']}").status_code == 404
    assert bia.get(f"/training/jobs/{job['id']}/stream").status_code == 404
    assert bia.post(f"/training/jobs/{job['id']}/publish", json={"name": "x"}).status_code == 404

    for _ in range(200):
        if ana.get(f"/training/jobs/{job['id']}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    model = ana.post(f"/training/jobs/{job['id']}/publish", json={"name": "m"}).json()
    assert bia.get(f"/models/{model['id']}/download").status_code == 404
    assert bia.get(f"/datasets/{dataset['id']}/download").status_code == 404
    assert ana.get(f"/models/{model['id']}/download").status_code == 200


def test_segmentation_and_imports_are_private(app_factory):
    from .test_dataset_and_training import _make_annotated_project
    from .test_import import DAT, _import_dat
    from .test_segmentation import _noise_image_bytes, _publish_model, _wait

    ana, bia = app_factory(), app_factory()
    register(ana)
    register(bia, email="bia@example.com", name="Bia")

    project, image = _make_annotated_project(ana)
    model = _publish_model(ana, project)
    run = ana.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).json()
    _wait(ana, f"/segmentation/runs/{run['id']}")
    [result] = ana.get(f"/segmentation/runs/{run['id']}/results").json()

    # bia can't start runs on ana's model/images, nor read or delete ana's runs
    assert bia.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).status_code == 404
    bia_project = bia.post("/projects", json={"name": "b"}).json()
    assert bia.post("/segmentation/runs", json={"model_id": "x", "image_ids": [image["id"]]}).status_code == 404
    for resp in [
        bia.get(f"/projects/{project['id']}/runs"),
        bia.get(f"/segmentation/runs/{run['id']}"),
        bia.get(f"/segmentation/runs/{run['id']}/stream"),
        bia.get(f"/segmentation/runs/{run['id']}/results"),
        bia.get(f"/segmentation/runs/{run['id']}/export"),
        bia.delete(f"/segmentation/runs/{run['id']}"),
        bia.get(f"/segmentation/results/{result['id']}/bin"),
        bia.get(f"/segmentation/results/{result['id']}/overlay"),
        _import_dat(bia, project),
    ]:
        assert resp.status_code == 404, resp.request.url
    # importing into her own project with ana's dataset is also refused
    dataset = ana.get(f"/projects/{project['id']}/datasets").json()[0]
    resp = bia.post(
        f"/projects/{bia_project['id']}/models/import",
        data={"name": "n", "dataset_id": dataset["id"]},
        files={"file": ("m.pt", b"x")},
    )
    assert resp.status_code == 404
    assert ana.get(f"/segmentation/runs/{run['id']}/export").status_code == 200


def test_create_user_script(api_client, capsys):
    from app.create_user import main

    assert main(["novo@example.com", "--name", "Novo", "--password", "senha-forte-123"]) == 0
    assert main(["novo@example.com", "--password", "senha-forte-123"]) == 1  # duplicate
    assert main(["ruim", "--password", "senha-forte-123"]) == 1  # invalid email
    assert main(["b@example.com", "--password", "curta"]) == 1  # short password
    api_client.post("/auth/logout")
    resp = api_client.post("/auth/login", json={"email": "novo@example.com", "password": "senha-forte-123"})
    assert resp.status_code == 200 and resp.json()["name"] == "Novo"
