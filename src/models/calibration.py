"""Temperature scaling for post-hoc confidence calibration.

Fits a single scalar T > 0 on a held-out calibration split by minimizing
NLL of logits / T against true labels. Does not change the argmax
prediction (and therefore accuracy), only the sharpness of the softmax
distribution -- fixing the overconfident-wrong-prediction problem.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        # Clamp away from 0 to avoid a degenerate, near-infinitely-sharp
        # softmax if LBFGS drives log_temperature to a large negative value.
        return self.log_temperature.exp().clamp(min=1e-3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """Fits log_temperature via LBFGS on the given (logits, labels) batch.

        Intended to be called once on the full calibration split (not
        mini-batched), since LBFGS assumes a fixed objective across steps.
        Returns the post-fit cross-entropy loss.
        """
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return F.cross_entropy(self.forward(logits), labels).item()

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=-1)


def high_confidence_error_rate(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.8
) -> float:
    """Fraction of predictions with confidence >= threshold that are wrong."""
    confidences, predictions = probs.max(dim=-1)
    mask = confidences >= threshold
    if mask.sum().item() == 0:
        return 0.0
    wrong = (predictions[mask] != labels[mask]).float().mean().item()
    return wrong
