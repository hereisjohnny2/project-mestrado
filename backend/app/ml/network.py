"""Training/testing loops, ported verbatim from
``legacy/rock-nn/utils/network.py``.

The loss (``nll_loss``), optimizer step order and accuracy computation are
untouched — this is the code whose numerical behaviour the parity test
protects. Richer metrics live in :mod:`training`, computed *after* these
functions finish so they never perturb the RNG stream that determines the
trained weights.
"""

import torch
import torch.nn.functional as F


def run_training(num_epochs, train_dataloader, net, optimizer, on_epoch_end=None):
    """Identical to the legacy loop, with an optional progress callback.

    ``on_epoch_end(epoch_index, epochs, loss)`` is called after each epoch
    if provided — used to stream progress over SSE. It performs no tensor
    operations, so it cannot affect the RNG stream or the trained weights.
    """
    loss = None
    for epoch in range(num_epochs):
        for data in train_dataloader:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 3))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(f"{epoch + 1} of {num_epochs} epochs - Loss: {loss}")
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, num_epochs, float(loss.item()) if loss is not None else None)
    return loss


def run_test(test_dataloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            X, y = data
            output = net(X.view(-1, 3))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    return round(correct / total, 3)
