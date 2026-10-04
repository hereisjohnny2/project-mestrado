"""Segmentation inference, ported from ``legacy/rock-nn/utils/image.py``.

Kept identical: RGB image -> raw 0-255 float tensor -> per-pixel
``argmax`` over the model's two logits -> binary mask (1 = pore, 0 = solid).

Changed (infrastructure only, see plan §3.2):

- ``calculate_porosity`` is vectorized (``arr.mean()``) instead of a nested
  Python loop over every pixel.
- ``binarize`` processes the flattened pixel array in chunks instead of a
  single giant tensor, so large images don't blow up memory. For any image
  small enough to fit in one chunk (the default is 1,000,000 pixels) this is
  mathematically the same single call the legacy code made — chunking only
  changes how a large image is batched, not the per-pixel decision, since
  the model has no cross-pixel state (no batchnorm, no pooling).
"""

from __future__ import annotations

import time
from os import path

import numpy as np
import torch
from PIL import Image


def _device_for(net: torch.nn.Module) -> torch.device:
    return next(net.parameters()).device


def binarize(arr: np.ndarray, img_size: tuple[int, int], net, chunk_size: int = 1_000_000) -> np.ndarray:
    """Same semantics as the legacy ``binarize``: flatten to (N, 3), run the
    model, argmax, reshape back to (H, W). Runs in chunks to bound memory."""
    width, height = img_size
    device = _device_for(net)

    flat = torch.from_numpy(arr).reshape(-1, 3).float()
    n_pixels = flat.shape[0]

    out = torch.empty(n_pixels, dtype=torch.long)

    net.eval()
    with torch.inference_mode():
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            chunk = flat[start:end]
            if device.type != "cpu":
                chunk = chunk.to(device)
            preds = torch.argmax(net(chunk), dim=1)
            out[start:end] = preds.cpu()

    return out.view(height, width).numpy()


def predict_classes(arr: np.ndarray, img_size: tuple[int, int], net, architecture: str = "1.0.0", chunk_size: int = 1_000_000) -> np.ndarray:
    """Per-pixel class map. Architecture 1.0.0 is the legacy ``binarize``
    (0 = rest, 1 = pore); other architectures write the project's class
    index (output ``j`` -> class ``j + 1``, so 0 never appears)."""
    if architecture == "1.0.0":
        return binarize(arr, img_size, net, chunk_size)

    from .features import extract_features_v2

    width, height = img_size
    device = _device_for(net)
    flat_rgb = np.asarray(arr).reshape(-1, 3)
    n_pixels = flat_rgb.shape[0]
    out = torch.empty(n_pixels, dtype=torch.long)

    net.eval()
    with torch.inference_mode():
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            chunk = torch.from_numpy(extract_features_v2(flat_rgb[start:end]))
            if device.type != "cpu":
                chunk = chunk.to(device)
            out[start:end] = torch.argmax(net(chunk), dim=1).cpu() + 1

    return out.view(height, width).numpy()


def apply_binarization(image_path: str, net, architecture: str = "1.0.0") -> tuple[np.ndarray, float]:
    """Returns (class map array, elapsed milliseconds) — same return shape
    as the legacy ``@timing_decorator``-wrapped ``apply_binarization``; for
    architecture 1.0.0 the map is the legacy binary mask."""
    start = time.time()
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img)
    class_map = predict_classes(arr, img.size, net, architecture)
    elapsed_ms = (time.time() - start) * 1000
    return class_map, elapsed_ms


def calculate_porosity(arr: np.ndarray) -> float:
    """Fraction of pixels labelled pore (1). Vectorized replacement for the
    legacy nested-loop version — same value, same semantics."""
    return float(arr.mean())


RGB = tuple[int, int, int]
BINARY_PALETTE: dict[int, RGB] = {0: (0, 0, 0), 1: (255, 255, 255)}


def save_class_map(class_map: np.ndarray, output_path_without_suffix: str, palette: dict[int, RGB]) -> str:
    """Writes the per-pixel class indices as a paletted PNG: the file keeps
    the raw index per pixel (readable back with ``Image.open(...)``) while
    any viewer renders it with the class colors. ``BINARY_PALETTE`` gives
    the legacy black/white binarized image."""
    img = Image.fromarray(class_map.astype(np.uint8), mode="P")
    flat = [0] * (256 * 3)
    for index, (r, g, b) in palette.items():
        flat[index * 3 : index * 3 + 3] = [r, g, b]
    img.putpalette(flat)
    out_path = f"{output_path_without_suffix}-bin.png"
    img.save(out_path)
    return out_path


def save_overlay(original_path: str, class_map: np.ndarray, output_path_without_suffix: str, colors: dict[int, RGB]) -> str:
    """Extra (not in the legacy app): the original image with every class
    painted in its color as a translucent overlay, used by the segmentation
    gallery. Index 0 (and any index without a color) is left untouched."""
    base = Image.open(original_path).convert("RGB")
    overlay = np.array(base).copy()
    for index, color in colors.items():
        pixels = class_map == index
        for c in range(3):
            overlay[..., c] = np.where(pixels, color[c], overlay[..., c])
    blended = Image.blend(base, Image.fromarray(overlay), alpha=0.4)
    out_path = f"{output_path_without_suffix}-overlay.png"
    blended.save(out_path)
    return out_path


def class_fractions(class_map: np.ndarray, index_to_name: dict[int, str]) -> dict[str, float]:
    return {name: float((class_map == index).mean()) for index, name in index_to_name.items()}


def save_porosity(porosity: float, name: str, file_path: str, elapsed_ms: float) -> None:
    """Same line format as the legacy ``porosity.txt``: name, porosity, ms."""
    with open(file_path, "a") as f:
        f.write(f"{name} \t {porosity} \t {elapsed_ms}\n")


def convert_image_to_rgb_format(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    name, _ = path.splitext(image_path)
    out_path = f"{name}-converted.png"
    img.save(out_path)
    return out_path
