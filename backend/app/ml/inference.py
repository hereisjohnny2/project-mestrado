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


def apply_binarization(image_path: str, net) -> tuple[np.ndarray, float]:
    """Returns (binary mask array, elapsed milliseconds) — same return shape
    as the legacy ``@timing_decorator``-wrapped ``apply_binarization``."""
    start = time.time()
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img)
    mask = binarize(arr, img.size, net)
    elapsed_ms = (time.time() - start) * 1000
    return mask, elapsed_ms


def calculate_porosity(arr: np.ndarray) -> float:
    """Fraction of pixels labelled pore (1). Vectorized replacement for the
    legacy nested-loop version — same value, same semantics."""
    return float(arr.mean())


def save_image(arr: np.ndarray, output_path_without_suffix: str) -> str:
    out_path = f"{output_path_without_suffix}-bin.png"
    Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)
    return out_path


def save_overlay(original_path: str, mask: np.ndarray, output_path_without_suffix: str, pore_color=(255, 80, 80)) -> str:
    """Extra (not in the legacy app): the original image with the pore mask
    painted as a translucent overlay, used by the segmentation gallery."""
    base = Image.open(original_path).convert("RGB")
    overlay = np.array(base).copy()
    pore_pixels = mask.astype(bool)
    for c in range(3):
        overlay[..., c] = np.where(pore_pixels, pore_color[c], overlay[..., c])
    blended = Image.blend(base, Image.fromarray(overlay), alpha=0.4)
    out_path = f"{output_path_without_suffix}-overlay.png"
    blended.save(out_path)
    return out_path


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
