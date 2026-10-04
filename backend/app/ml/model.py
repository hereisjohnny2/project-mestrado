"""Neural network architecture for pore/solid pixel classification.

This is a byte-for-byte port of ``legacy/rock-nn/rock_model.py``. The
architecture, activations and output are intentionally left untouched so
that a model trained here is numerically identical to one trained with the
original CLI (see ``backend/tests/test_parity.py``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RockNetModel(nn.Module):
    """MLP classifier: 3 RGB channels in, 2 classes out (solid, pore)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class RockNetV2(nn.Module):
    """Architecture 2.0.0: ``n_features -> W -> W/2 -> n_classes`` MLP over the
    normalized color attributes of :mod:`features`. Same ``log_softmax``
    output convention as ``RockNetModel`` so inference stays an ``argmax``."""

    def __init__(self, n_features: int = 8, hidden_width: int = 32, n_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_width)
        self.fc2 = nn.Linear(hidden_width, max(hidden_width // 2, 2))
        self.fc3 = nn.Linear(max(hidden_width // 2, 2), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
