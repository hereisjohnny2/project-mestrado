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


@dataclass
class TrainingResult:
    model: RockNetModel
    accuracy: float
    loss_curve: list[float]
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]], class 1 = pore
    per_class: dict
    duration_ms: float
    config: TrainingConfig


def _confusion_and_per_class(test_dataloader, net) -> tuple[list[list[int]], dict]:
    """Extra metrics pass — read-only, no grad, no RNG consumption (the test
    loader is never shuffled), run after training/testing complete."""
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            output = net(X.view(-1, 3))
            preds = torch.argmax(output, dim=1)
            for pred, label in zip(preds.tolist(), y.tolist()):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
                else:
                    tn += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    precision_pore = safe_div(tp, tp + fp)
    recall_pore = safe_div(tp, tp + fn)
    f1_pore = safe_div(2 * precision_pore * recall_pore, precision_pore + recall_pore)
    iou_pore = safe_div(tp, tp + fp + fn)

    precision_solid = safe_div(tn, tn + fn)
    recall_solid = safe_div(tn, tn + fp)
    f1_solid = safe_div(2 * precision_solid * recall_solid, precision_solid + recall_solid)
    iou_solid = safe_div(tn, tn + fp + fn)

    per_class = {
        "pore": {"precision": precision_pore, "recall": recall_pore, "f1": f1_pore, "iou": iou_pore},
        "solid": {"precision": precision_solid, "recall": recall_solid, "f1": f1_solid, "iou": iou_solid},
    }
    confusion_matrix = [[tn, fp], [fn, tp]]
    return confusion_matrix, per_class


def train(
    dataset_path: str,
    config: TrainingConfig | None = None,
    on_epoch_end=None,
) -> TrainingResult:
    """Train a RockNetModel from a ``.dat`` file.

    With default ``config`` and the caller having called
    ``torch.manual_seed(seed)`` beforehand (or ``config.seed`` set), this
    reproduces the legacy CLI's ``trainer.train_from_dataset`` bit-for-bit.
    """
    config = config or TrainingConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)

    start = time.time()

    # --- order below matches legacy trainer.py exactly ---
    train_dataloader, test_dataloader = create_dataloaders(
        dataset_path, ratio=config.split_ratio, batch_size=config.batch_size
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

    confusion_matrix, per_class = _confusion_and_per_class(test_dataloader, net)

    return TrainingResult(
        model=net,
        accuracy=accuracy,
        loss_curve=loss_curve,
        confusion_matrix=confusion_matrix,
        per_class=per_class,
        duration_ms=duration_ms,
        config=config,
    )
