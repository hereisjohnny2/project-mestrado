"""Per-pixel color attributes for architecture 2.0.0.

Everything here is a pointwise function of a pixel's RGB — no neighborhood
is involved — so the same vector can be computed from a ``.dat`` row at
training time and from an image pixel at inference time. Color conversions
are written in numpy on purpose (no scikit-image dependency); they assume
sRGB input with a D65 white point.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES_V2 = ["R", "G", "B", "S", "V", "L", "a", "b"]

# sRGB (linear) -> XYZ, D65 reference white.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6 / 29


def rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    """``(N, 3)`` RGB in [0, 1] -> ``(N, 3)`` H in [0, 1), S and V in [0, 1]."""
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]
    v = rgb01.max(axis=1)
    delta = v - rgb01.min(axis=1)
    s = np.divide(delta, v, out=np.zeros_like(v), where=v > 0)

    h = np.zeros_like(v)
    nonzero = delta > 0
    safe_delta = np.where(nonzero, delta, 1.0)
    is_r = nonzero & (v == r)
    is_g = nonzero & ~is_r & (v == g)
    is_b = nonzero & ~is_r & ~is_g
    h[is_r] = ((g - b) / safe_delta)[is_r] % 6
    h[is_g] = ((b - r) / safe_delta)[is_g] + 2
    h[is_b] = ((r - g) / safe_delta)[is_b] + 4
    h = h / 6
    return np.stack([h, s, v], axis=1)


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """``(N, 3)`` sRGB in [0, 1] -> ``(N, 3)`` CIE L*a*b* (L in [0, 100])."""
    c = np.asarray(rgb01, dtype=np.float64)
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _D65
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4 / 29)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def extract_features_v2(rgb_uint8: np.ndarray) -> np.ndarray:
    """``(N, 3)`` uint8 RGB -> ``(N, 8)`` float32 feature rows, every column
    scaled to roughly [0, 1]: R, G, B, S, V, L, a, b (see FEATURE_NAMES_V2)."""
    rgb01 = np.asarray(rgb_uint8).reshape(-1, 3).astype(np.float64) / 255.0
    hsv = rgb_to_hsv(rgb01)
    lab = rgb_to_lab(rgb01)
    features = np.concatenate(
        [
            rgb01,
            hsv[:, 1:3],
            np.stack([lab[:, 0] / 100.0, (lab[:, 1] + 128) / 255.0, (lab[:, 2] + 128) / 255.0], axis=1),
        ],
        axis=1,
    )
    return features.astype(np.float32)
