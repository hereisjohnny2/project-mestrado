"""Fase 1 acceptance criterion (plan §6): annotate an image, reload, the
annotation is still there — exercised here at the API level."""

import io

from PIL import Image as PILImage


def _png_bytes(size=(8, 6), color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_project_and_image_upload_round_trip(api_client):
    resp = api_client.post("/projects", json={"name": "Amostra 1"})
    assert resp.status_code == 201
    project = resp.json()
    assert project["name"] == "Amostra 1"

    files = [("files", ("rock.png", _png_bytes(), "image/png"))]
    resp = api_client.post(f"/projects/{project['id']}/images", files=files)
    assert resp.status_code == 201
    images = resp.json()
    assert len(images) == 1
    image = images[0]
    assert image["width"] == 8
    assert image["height"] == 6
    assert image["has_annotation"] is False

    resp = api_client.get(f"/projects/{project['id']}")
    assert resp.status_code == 200
    assert len(resp.json()["images"]) == 1

    resp = api_client.get(f"/images/{image['id']}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_mask_persists_across_reload(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("rock.png", _png_bytes(), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]

    resp = api_client.get(f"/images/{image['id']}/mask")
    assert resp.status_code == 404

    mask_bytes = _png_bytes(size=(8, 6), color=(1, 1, 1))
    resp = api_client.put(f"/images/{image['id']}/mask", content=mask_bytes, headers={"content-type": "image/png"})
    assert resp.status_code == 200
    assert resp.json()["has_annotation"] is True

    resp = api_client.get(f"/images/{image['id']}/mask")
    assert resp.status_code == 200
    assert resp.content == mask_bytes

    updated_mask = _png_bytes(size=(8, 6), color=(2, 2, 2))
    resp = api_client.put(f"/images/{image['id']}/mask", content=updated_mask, headers={"content-type": "image/png"})
    assert resp.status_code == 200

    resp = api_client.get(f"/images/{image['id']}/mask")
    assert resp.content == updated_mask


def test_unsupported_image_rejected(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("bad.txt", b"not an image", "text/plain"))]
    resp = api_client.post(f"/projects/{project['id']}/images", files=files)
    assert resp.status_code == 422


def test_rename_project(api_client):
    project = api_client.post("/projects", json={"name": "old name"}).json()
    resp = api_client.patch(f"/projects/{project['id']}", json={"name": "new name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new name"
    assert api_client.get(f"/projects/{project['id']}").json()["name"] == "new name"


def test_delete_project_removes_images(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("rock.png", _png_bytes(), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]

    resp = api_client.delete(f"/projects/{project['id']}")
    assert resp.status_code == 204
    assert api_client.get(f"/projects/{project['id']}").status_code == 404
    assert api_client.get(f"/images/{image['id']}").status_code == 404


def test_delete_image(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("rock.png", _png_bytes(), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]

    resp = api_client.delete(f"/images/{image['id']}")
    assert resp.status_code == 204
    assert api_client.get(f"/images/{image['id']}").status_code == 404
    assert api_client.get(f"/projects/{project['id']}").json()["images"] == []


def test_clear_image_mask(api_client):
    project = api_client.post("/projects", json={"name": "p"}).json()
    files = [("files", ("rock.png", _png_bytes(), "image/png"))]
    image = api_client.post(f"/projects/{project['id']}/images", files=files).json()[0]

    mask_bytes = _png_bytes(size=(8, 6), color=(1, 1, 1))
    api_client.put(f"/images/{image['id']}/mask", content=mask_bytes, headers={"content-type": "image/png"})

    resp = api_client.delete(f"/images/{image['id']}/mask")
    assert resp.status_code == 200
    assert resp.json()["has_annotation"] is False
    assert api_client.get(f"/images/{image['id']}/mask").status_code == 404
