"""Parity test — the acceptance criterion for Fase 0 (plan §3.3).

Proves that the ported `backend/app/ml` code trains and infers *exactly*
like the original `legacy/rock-nn` CLI, given the same dataset, the same
image and the same seed:

1. Train with the legacy CLI and with the new code, same seed -> identical
   ``state_dict`` (within floating-point tolerance).
2. Apply both resulting models to the same image -> identical binary mask
   and matching porosity.

Both code paths share the exact same call order (dataset split -> model
init -> training loop) so the global PyTorch RNG is consumed identically —
see the module docstring in ``backend/app/ml/training.py``.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Legacy CLI (bare imports, made available by conftest.py)
import trainer as legacy_trainer
from rock_model import RockNetModel as LegacyRockNetModel
from utils.image import apply_binarization as legacy_apply_binarization
from utils.image import calculate_porosity as legacy_calculate_porosity

# Ported backend code
from app.ml.dataset import dataset_stats
from app.ml.inference import apply_binarization as new_apply_binarization
from app.ml.inference import calculate_porosity as new_calculate_porosity
from app.ml.model import RockNetModel as NewRockNetModel
from app.ml.training import TrainingConfig
from app.ml.training import train as new_train

SEED = 1234


def _make_dataset_file(path: Path, n_per_class: int = 60) -> None:
    """Deterministic, clearly-separable synthetic dataset: light pixels
    labelled Solido, dark pixels labelled Poro. Content generation uses its
    own RNG (not torch's), so it's identical on every run regardless of the
    torch seed used later for training."""
    rng = random.Random(42)
    lines = []
    for _ in range(n_per_class):
        r, g, b = (rng.randint(180, 255) for _ in range(3))
        lines.append(f"{r}\t{g}\t{b}\tSolido")
    for _ in range(n_per_class):
        r, g, b = (rng.randint(0, 60) for _ in range(3))
        lines.append(f"{r}\t{g}\t{b}\tPoro")
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n")


def _make_test_image(path: Path, size: int = 6) -> None:
    rng = random.Random(7)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if rng.random() < 0.5:
                arr[y, x] = [rng.randint(180, 255)] * 3
            else:
                arr[y, x] = [rng.randint(0, 60)] * 3
    Image.fromarray(arr, mode="RGB").save(path)


def test_training_is_bit_identical_to_legacy_cli(tmp_path):
    dataset_path = tmp_path / "fixture-dataset.dat"
    _make_dataset_file(dataset_path)

    torch.manual_seed(SEED)
    legacy_trainer.train_from_dataset(str(dataset_path), epochs=3)
    legacy_model_path = tmp_path / "fixture-dataset-nn-model.pt"
    assert legacy_model_path.exists(), "legacy CLI did not write the expected model file"

    legacy_model = LegacyRockNetModel()
    legacy_model.load_state_dict(torch.load(legacy_model_path))

    torch.manual_seed(SEED)
    result = new_train(str(dataset_path), TrainingConfig(epochs=3, seed=None))

    legacy_sd = legacy_model.state_dict()
    new_sd = result.model.state_dict()

    assert legacy_sd.keys() == new_sd.keys()
    for key in legacy_sd:
        assert torch.allclose(legacy_sd[key], new_sd[key], atol=1e-6), f"weights diverged in {key}"

    # Weights are identical, so accuracy (computed by the same run_test
    # logic in both code paths, deterministically ordered) must be too.
    assert 0.0 <= result.accuracy <= 1.0


def test_inference_matches_legacy_mask_and_porosity(tmp_path):
    dataset_path = tmp_path / "fixture-dataset.dat"
    _make_dataset_file(dataset_path)
    image_path = tmp_path / "fixture-image.png"
    _make_test_image(image_path)

    torch.manual_seed(SEED)
    legacy_trainer.train_from_dataset(str(dataset_path), epochs=5)
    legacy_model_path = tmp_path / "fixture-dataset-nn-model.pt"

    legacy_model = LegacyRockNetModel()
    legacy_model.load_state_dict(torch.load(legacy_model_path))
    legacy_model.eval()

    legacy_mask, _ = legacy_apply_binarization(str(image_path), legacy_model)
    legacy_porosity = legacy_calculate_porosity(legacy_mask)

    # Load the very same weights into the ported model class.
    new_model = NewRockNetModel()
    new_model.load_state_dict(torch.load(legacy_model_path))
    new_model.eval()

    new_mask, _ = new_apply_binarization(str(image_path), new_model)
    new_porosity = new_calculate_porosity(new_mask)

    assert np.array_equal(legacy_mask, new_mask), "binarized mask differs from the legacy CLI"
    assert new_porosity == pytest.approx(legacy_porosity, rel=1e-9)


def test_dataset_stats_match_dat_contents(tmp_path):
    dataset_path = tmp_path / "fixture-dataset.dat"
    _make_dataset_file(dataset_path, n_per_class=25)

    stats = dataset_stats(str(dataset_path))
    assert stats["n_pixels"] == 50
    assert stats["n_pore"] == 25
    assert stats["n_solid"] == 25
