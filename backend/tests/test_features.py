"""Color attribute extraction (architecture 2.0.0) against reference values
of the standard sRGB/D65 conversions."""

import numpy as np
import pytest

from app.ml.features import FEATURE_NAMES_V2, extract_features_v2, rgb_to_hsv, rgb_to_lab


def test_hsv_reference_values():
    rgb = np.array([[255, 0, 0], [128, 128, 128], [0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]) / 255.0
    hsv = rgb_to_hsv(rgb)
    # red
    assert hsv[0, 1] == pytest.approx(1.0) and hsv[0, 2] == pytest.approx(1.0) and hsv[0, 0] == pytest.approx(0.0)
    # gray: no saturation, V = 128/255
    assert hsv[1, 1] == pytest.approx(0.0) and hsv[1, 2] == pytest.approx(0.502, abs=1e-3)
    # black: S and V are 0 (no division by zero)
    assert hsv[2, 1] == 0.0 and hsv[2, 2] == 0.0
    # green / blue hues at 1/3 and 2/3 of the circle
    assert hsv[3, 0] == pytest.approx(1 / 3) and hsv[4, 0] == pytest.approx(2 / 3)
    assert hsv[5, 1] == pytest.approx(0.0) and hsv[5, 2] == pytest.approx(1.0)


def test_lab_reference_values():
    rgb = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]) / 255.0
    lab = rgb_to_lab(rgb)
    expected = [
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (53.24, 80.09, 67.20),
        (87.74, -86.18, 83.18),
        (32.30, 79.19, -107.86),
    ]
    for row, (l_ref, a_ref, b_ref) in zip(lab, expected):
        assert row[0] == pytest.approx(l_ref, abs=0.05)
        assert row[1] == pytest.approx(a_ref, abs=0.05)
        assert row[2] == pytest.approx(b_ref, abs=0.05)


def test_feature_vector_shape_and_range():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)
    rgb = np.vstack([rgb, [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]])
    feats = extract_features_v2(rgb)
    assert feats.shape == (rgb.shape[0], len(FEATURE_NAMES_V2))
    assert feats.dtype == np.float32
    assert feats.min() >= 0.0 and feats.max() <= 1.0
    # RGB columns are just the input scaled
    assert np.allclose(feats[:, :3], rgb / 255.0, atol=1e-6)
    # an (H, W, 3) image flattens the same way as (N, 3)
    assert np.array_equal(extract_features_v2(rgb.reshape(-1, 2, 3)), feats)
