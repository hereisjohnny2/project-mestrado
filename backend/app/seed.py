"""Seeds a running backend with a demo project, so a fresh install has
something to click on: synthetic "rock" images (dark pores on a lighter,
grainy solid), a pre-painted annotation mask for each, a generated dataset,
and one trained + published model.

    python -m app.seed [--url http://localhost:8000]

Everything goes through the public HTTP API, so it also works from the host
against ``docker compose up`` (``docker compose exec backend python -m app.seed``).
The images are generated, not real thin sections — they exist to exercise
the workflow, not to say anything about porosity.
"""

from __future__ import annotations

import argparse
import io
import time

import httpx
import numpy as np
from PIL import Image

SIZE = 128
N_IMAGES = 4
PROJECT_NAME = "Exemplo (dados sintéticos)"


def synthetic_rock(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns an RGB image and a boolean pore map (True = pore)."""
    rng = np.random.default_rng(seed)
    # Smooth noise: white noise low-passed by repeated box blurs, thresholded.
    field = rng.random((SIZE, SIZE))
    for _ in range(6):
        field = (
            field
            + np.roll(field, 1, 0)
            + np.roll(field, -1, 0)
            + np.roll(field, 1, 1)
            + np.roll(field, -1, 1)
        ) / 5
    pore = field < np.quantile(field, 0.3)

    solid_rgb = np.array([185, 170, 150])
    pore_rgb = np.array([45, 55, 70])
    img = np.where(pore[..., None], pore_rgb, solid_rgb) + rng.normal(0, 12, (SIZE, SIZE, 3))
    return np.clip(img, 0, 255).astype(np.uint8), pore


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def partial_mask(pore: np.ndarray, seed: int) -> np.ndarray:
    """Annotates a random ~15% of pixels, like a user painting a few strokes.
    Values follow the mask format: 0 = unannotated, 1 = poro, 2 = solido."""
    rng = np.random.default_rng(seed)
    picked = rng.random(pore.shape) < 0.15
    mask = np.zeros(pore.shape, dtype=np.uint8)
    mask[picked & pore] = 1
    mask[picked & ~pore] = 2
    return mask


def wait_for(client: httpx.Client, path: str) -> dict:
    deadline = time.time() + 300
    while time.time() < deadline:
        body = client.get(path).json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.5)
    raise TimeoutError(path)


def seed(client: httpx.Client) -> None:
    if any(p["name"] == PROJECT_NAME for p in client.get("/projects").json()):
        print(f"Projeto '{PROJECT_NAME}' já existe — nada a fazer.")
        return

    project = client.post("/projects", json={"name": PROJECT_NAME}).raise_for_status().json()
    pores = {}
    files = []
    for i in range(N_IMAGES):
        rgb, pore = synthetic_rock(i)
        pores[f"amostra-{i + 1}.png"] = pore
        files.append(("files", (f"amostra-{i + 1}.png", png_bytes(rgb), "image/png")))
    images = client.post(f"/projects/{project['id']}/images", files=files).raise_for_status().json()

    # Only the first half is annotated; the rest are left for the
    # segmentation step, as in real use.
    for i, image in enumerate(images[: N_IMAGES // 2]):
        mask = partial_mask(pores[image["filename"]], seed=100 + i)
        client.put(
            f"/images/{image['id']}/mask", content=png_bytes(mask), headers={"content-type": "image/png"}
        ).raise_for_status()

    dataset = client.post(f"/projects/{project['id']}/datasets").raise_for_status().json()
    print(f"Dataset: {dataset['n_pixels']} pixels ({dataset['n_pore']} poro, {dataset['n_solid']} sólido)")

    job = client.post(
        "/training/jobs", json={"dataset_id": dataset["id"], "epochs": 5, "seed": 0}
    ).raise_for_status().json()
    job = wait_for(client, f"/training/jobs/{job['id']}")
    if job["status"] != "done":
        raise SystemExit(f"Treino falhou: {job['error']}")
    model = client.post(f"/training/jobs/{job['id']}/publish", json={"name": "exemplo"}).raise_for_status().json()
    print(f"Modelo publicado: acurácia {model['metrics']['accuracy']:.3f}")
    print(f"Pronto. Abra o projeto '{PROJECT_NAME}' no frontend e rode a segmentação nas imagens não anotadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        seed(client)


if __name__ == "__main__":
    main()
