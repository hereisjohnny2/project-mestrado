"""Architecture 2.0.0: color attributes in, one output per project class."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from app.ml.architectures import ARCHITECTURES, build_model, describe
from app.ml.artifacts import load_metadata, load_model, save_model
from app.ml.inference import predict_classes
from app.ml.training import TrainingConfig, train

from .test_classes import THREE, _three_class_project
from .test_segmentation import _wait

CLASSES = ["Poro", "Sólido", "Cimento"]


def _three_class_dat(path: Path, n_per_class: int = 80) -> None:
    """Dark gray = Poro, light beige = Sólido, saturated green = Cimento:
    separable by brightness and saturation, which is what 2.0.0 sees."""
    rng = random.Random(1)
    lines = []
    for _ in range(n_per_class):
        lines.append(f"{rng.randint(0, 50)}\t{rng.randint(0, 50)}\t{rng.randint(0, 50)}\tPoro")
        lines.append(f"{rng.randint(190, 230)}\t{rng.randint(175, 215)}\t{rng.randint(150, 190)}\tSólido")
        lines.append(f"{rng.randint(0, 60)}\t{rng.randint(170, 255)}\t{rng.randint(0, 60)}\tCimento")
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n")


def test_registry_describes_both_architectures():
    assert set(ARCHITECTURES) == {"1.0.0", "2.0.0"}
    v1 = describe("1.0.0", n_classes=3)
    assert v1["layers"] == [3, 4, 4, 4, 2] and v1["supports_multiclass"] is False
    v2 = describe("2.0.0", n_classes=3)
    assert v2["layers"] == [8, 32, 16, 3] and v2["hidden_width"] == 32
    assert describe("2.0.0", n_classes=2, hidden_width=8)["layers"] == [8, 8, 4, 2]


def test_v2_trains_separable_classes_and_round_trips(tmp_path):
    dat = tmp_path / "three.dat"
    _three_class_dat(dat)

    config = TrainingConfig(epochs=30, batch_size=16, learning_rate=0.01, seed=7, architecture="2.0.0", hidden_width=16)
    result = train(str(dat), config, class_names=CLASSES)
    assert result.class_names == CLASSES
    assert result.accuracy >= 0.95, result.per_class
    assert len(result.confusion_matrix) == 3 and set(result.per_class) == set(CLASSES)
    assert len(result.loss_curve) == 30

    paths = save_model(result, tmp_path / "models", "v2", dataset_sha256="x")
    assert Path(paths["scripted_path"]).exists()
    meta = load_metadata(paths["json_path"])
    assert meta["format_version"] == 2
    assert meta["architecture"] == "2.0.0" and meta["hidden_width"] == 16
    assert meta["class_names"] == CLASSES and meta["feature_names"][-1] == "b"

    loaded = load_model(paths["pt_path"], architecture="2.0.0", n_classes=3, hidden_width=16)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    original = predict_classes(img, (7, 5), result.model, "2.0.0")
    assert np.array_equal(original, predict_classes(img, (7, 5), loaded, "2.0.0"))
    assert set(np.unique(original)) <= {1, 2, 3}  # project class indices, never 0

    # a wrong width cannot silently load the weights
    try:
        load_model(paths["pt_path"], architecture="2.0.0", n_classes=3, hidden_width=32)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected a shape mismatch")


def test_v2_rejects_labels_outside_the_class_list(tmp_path):
    dat = tmp_path / "three.dat"
    _three_class_dat(dat, n_per_class=5)
    try:
        train(str(dat), TrainingConfig(epochs=1, architecture="2.0.0"), class_names=["Poro", "Sólido"])
    except ValueError as exc:
        assert "Cimento" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_build_model_is_scriptable():
    scripted = torch.jit.script(build_model("2.0.0", n_classes=4, hidden_width=8).eval())
    out = scripted(torch.rand(3, 8))
    assert out.shape == (3, 4)


def test_api_lists_architectures_and_trains_v2(api_client):
    project, image = _three_class_project(api_client)
    dataset = api_client.post(f"/projects/{project['id']}/datasets").json()

    resp = api_client.get("/architectures", params={"n_classes": 3, "hidden_width": 8})
    assert resp.status_code == 200
    by_id = {a["id"]: a for a in resp.json()}
    assert by_id["1.0.0"]["layers"] == [3, 4, 4, 4, 2]
    assert by_id["2.0.0"]["layers"] == [8, 8, 4, 3]
    assert "Lab" in by_id["2.0.0"]["description"]

    assert api_client.post("/training/jobs", json={"dataset_id": dataset["id"], "architecture": "3.0.0"}).status_code == 422
    assert api_client.post("/training/jobs", json={"dataset_id": dataset["id"], "architecture": "2.0.0", "hidden_width": 1}).status_code == 422

    job = api_client.post(
        "/training/jobs",
        json={"dataset_id": dataset["id"], "architecture": "2.0.0", "hidden_width": 8, "epochs": 2, "batch_size": 16, "seed": 1},
    ).json()
    assert job["architecture"] == "2.0.0"
    assert _wait(api_client, f"/training/jobs/{job['id']}")["status"] == "done"

    model = api_client.post(f"/training/jobs/{job['id']}/publish", json={"name": "v2"}).json()
    assert model["architecture"] == "2.0.0"
    assert model["config"]["class_names"] == ["Poro", "Sólido", "Cimento"]
    assert model["config"]["hidden_width"] == 8
    assert set(model["metrics"]["per_class"]) == {"Poro", "Sólido", "Cimento"}
    meta = json.loads(api_client.get(f"/models/{model['id']}/download?fmt=json").content)
    assert meta["architecture"] == "2.0.0"

    run = api_client.post("/segmentation/runs", json={"model_id": model["id"], "image_ids": [image["id"]]}).json()
    assert _wait(api_client, f"/segmentation/runs/{run['id']}")["status"] == "done"
    result = api_client.get(f"/segmentation/runs/{run['id']}/results").json()[0]
    assert set(result["class_fractions"]) == {"Poro", "Sólido", "Cimento"}
    assert abs(sum(result["class_fractions"].values()) - 1.0) < 1e-9
    assert result["porosity"] == result["class_fractions"]["Poro"]

    bin_img = PILImage.open(io.BytesIO(api_client.get(f"/segmentation/results/{result['id']}/bin").content))
    assert bin_img.mode == "P"
    values = set(np.unique(np.asarray(bin_img)))
    assert values and values <= {1, 2, 3}
    # the palette carries the class colors (index 3 = Cimento = #16a34a)
    palette = bin_img.getpalette()
    assert palette[9:12] == [0x16, 0xA3, 0x4A]
    assert api_client.get(f"/segmentation/results/{result['id']}/overlay").status_code == 200

    csv_header = api_client.get(f"/segmentation/runs/{run['id']}/export").text.splitlines()[0]
    assert csv_header == "nome,porosidade,tempo_ms,Poro,Sólido,Cimento"
