"""Smoke test for the artifact save/load round trip (new functionality —
the legacy CLI never wrote metrics or a scripted model, see plan §1.3
item 3, so there's no legacy behaviour to compare against here)."""

from pathlib import Path

import torch

from app.ml.artifacts import load_metadata, load_model, save_model
from app.ml.training import TrainingConfig, train

DATASET_CONTENT = "\n".join(
    ["200\t200\t200\tSolido"] * 20 + ["20\t20\t20\tPoro"] * 20
)


def test_save_and_load_model_round_trip(tmp_path):
    dataset_path = tmp_path / "toy.dat"
    dataset_path.write_text(DATASET_CONTENT + "\n")

    torch.manual_seed(1)
    result = train(str(dataset_path), TrainingConfig(epochs=2))

    artifact_dir = tmp_path / "models"
    paths = save_model(result, artifact_dir, name="toy-model", dataset_sha256="deadbeef")

    for key in ("pt_path", "json_path", "scripted_path"):
        assert Path(paths[key]).exists()

    loaded = load_model(paths["pt_path"])
    for p1, p2 in zip(result.model.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)

    metadata = load_metadata(paths["json_path"])
    assert metadata["dataset_sha256"] == "deadbeef"
    assert metadata["config"]["epochs"] == 2
    assert "accuracy" in metadata["metrics"]
    assert 0.0 <= metadata["metrics"]["accuracy"] <= 1.0
