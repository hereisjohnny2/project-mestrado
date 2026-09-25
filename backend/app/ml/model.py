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
