"""Training orchestration.

``train()`` reproduces the exact call order of
``legacy/rock-nn/trainer.py::train_from_dataset`` — dataset split, then model
init, then training loop, then test — because that order determines how the
global PyTorch RNG is consumed. Keeping it identical is what makes the
parity test (``backend/tests/test_parity.py``) possible: seed the RNG the
same way before calling either code path and the resulting weights match
exactly.

Everything below that order (extra metrics) runs strictly *after* training
finishes, with no gradient and no randomness, so it cannot affect the
trained weights or break parity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.optim as optim

from .dataset import create_dataloaders
from .model import RockNetModel
from .network import run_test, run_training


@dataclass
class TrainingConfig:
    """Defaults match the legacy CLI exactly — the "padrão da dissertação"
    button on the training screen submits this unmodified."""

    epochs: int = 5
    learning_rate: float = 0.0025
    batch_size: int = 16
    split_ratio: float = 0.8
    seed: int | None = None
    architecture: str = "1.0.0"
    hidden_width: int | None = None  # architecture 2.0.0 only


BINARY_CLASS_NAMES = ["Sólido", "Poro"]  # output index 0 = not pore, 1 = pore


@dataclass
class TrainingResult:
    model: torch.nn.Module
    accuracy: float
    loss_curve: list[float]
    confusion_matrix: list[list[int]]  # rows = true class, cols = predicted, in class_names order
    per_class: dict
    duration_ms: float
    config: TrainingConfig
    class_names: list[str] = field(default_factory=lambda: list(BINARY_CLASS_NAMES))


def _confusion_and_per_class(test_dataloader, net, class_names: list[str]) -> tuple[list[list[int]], dict]:
    """Extra metrics pass — read-only, no grad, no RNG consumption (the test
    loader is never shuffled), run after training/testing complete."""
    k = len(class_names)
    confusion = [[0] * k for _ in range(k)]
    with torch.no_grad():
        for X, y in test_dataloader:
            preds = torch.argmax(net(X), dim=1)
            for pred, label in zip(preds.tolist(), y.tolist()):
                confusion[label][pred] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(k)) - tp
        fn = sum(confusion[i]) - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "iou": safe_div(tp, tp + fp + fn),
        }
    return confusion, per_class


def _accuracy(confusion: list[list[int]]) -> float:
    total = sum(map(sum, confusion))
    return sum(confusion[i][i] for i in range(len(confusion))) / total if total else 0.0


def evaluate_model(net: torch.nn.Module, dataset_path: str, class_names: list[str] | None = None) -> dict:
    """Metrics of an already-trained binary (architecture 1.0.0) ``net`` over
    a whole ``.dat`` file.

    Used for models imported from the legacy CLI, which saved no metrics:
    the numbers describe the model on *that* dataset (it may include the
    model's own training pixels, since the original split is unknown).
    """
    from torch.utils.data import DataLoader

    from .dataset import CustomDataset, load_data_from_file

    class_names = class_names or list(BINARY_CLASS_NAMES)
    start = time.time()
    loader = DataLoader(CustomDataset(load_data_from_file(dataset_path, class_names[1])), batch_size=1024, shuffle=False)
    net.eval()
    confusion_matrix, per_class = _confusion_and_per_class(loader, net, class_names)
    return {
        "accuracy": _accuracy(confusion_matrix),
        "loss_curve": [],
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
        "duration_ms": (time.time() - start) * 1000,
    }


def train(
    dataset_path: str,
    config: TrainingConfig | None = None,
    on_epoch_end=None,
    class_names: list[str] | None = None,
) -> TrainingResult:
    """Train a RockNetModel from a ``.dat`` file.

    With default ``config`` and the caller having called
    ``torch.manual_seed(seed)`` beforehand (or ``config.seed`` set), this
    reproduces the legacy CLI's ``trainer.train_from_dataset`` bit-for-bit.

    For architecture 1.0.0, ``class_names`` are the two output names,
    ``[not-pore, pore]``; the second is also the ``.dat`` label counted as
    pore. For other architectures they are the project's classes in index
    order, one output each.
    """
    config = config or TrainingConfig()
    if config.architecture != "1.0.0":
        if not class_names:
            raise ValueError("class_names are required for architecture " + config.architecture)
        return _train_v2(dataset_path, config, class_names, on_epoch_end)

    class_names = class_names or list(BINARY_CLASS_NAMES)
    if config.seed is not None:
        torch.manual_seed(config.seed)

    start = time.time()

    # --- order below matches legacy trainer.py exactly ---
    train_dataloader, test_dataloader = create_dataloaders(
        dataset_path, ratio=config.split_ratio, batch_size=config.batch_size, pore_label=class_names[1]
    )

    net = RockNetModel()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    loss_curve: list[float] = []

    def _capture(epoch_index, epochs, loss_value):
        if loss_value is not None:
            loss_curve.append(loss_value)
        if on_epoch_end is not None:
            on_epoch_end(epoch_index, epochs, loss_value)

    run_training(config.epochs, train_dataloader, net, optimizer, on_epoch_end=_capture)
    accuracy = run_test(test_dataloader, net)
    # --- end legacy-equivalent section ---

    duration_ms = (time.time() - start) * 1000

    confusion_matrix, per_class = _confusion_and_per_class(test_dataloader, net, class_names)

    return TrainingResult(
        model=net,
        accuracy=accuracy,
        loss_curve=loss_curve,
        confusion_matrix=confusion_matrix,
        per_class=per_class,
        duration_ms=duration_ms,
        config=config,
        class_names=class_names,
    )


def _train_v2(dataset_path: str, config: TrainingConfig, class_names: list[str], on_epoch_end=None) -> TrainingResult:
    """Architecture 2.0.0: color attributes in, one output per class. Not
    bound by the legacy call order, so it uses a plain TensorDataset."""
    import numpy as np
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from .architectures import build_model
    from .dataset import load_rows
    from .features import extract_features_v2

    if config.seed is not None:
        torch.manual_seed(config.seed)

    start = time.time()
    rows = load_rows(dataset_path)
    index_of = {name: i for i, name in enumerate(class_names)}
    unknown = sorted({row[3] for row in rows} - set(index_of))
    if unknown:
        raise ValueError("rótulos fora das classes do dataset: " + ", ".join(unknown))

    rgb = np.array([row[:3] for row in rows], dtype=np.uint8).reshape(-1, 3)
    X = torch.from_numpy(extract_features_v2(rgb))
    y = torch.tensor([index_of[row[3]] for row in rows], dtype=torch.long)

    dataset = TensorDataset(X, y)
    train_size = int(config.split_ratio * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=max(config.batch_size, 1024), shuffle=False)

    net = build_model(config.architecture, n_classes=len(class_names), hidden_width=config.hidden_width)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    loss_curve: list[float] = []
    for epoch in range(config.epochs):
        loss = None
        net.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        loss_value = float(loss.item()) if loss is not None else None
        if loss_value is not None:
            loss_curve.append(loss_value)
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, config.epochs, loss_value)

    net.eval()
    confusion_matrix, per_class = _confusion_and_per_class(test_loader, net, class_names)
    return TrainingResult(
        model=net,
        accuracy=round(_accuracy(confusion_matrix), 3),
        loss_curve=loss_curve,
        confusion_matrix=confusion_matrix,
        per_class=per_class,
        duration_ms=(time.time() - start) * 1000,
        config=config,
        class_names=class_names,
    )
