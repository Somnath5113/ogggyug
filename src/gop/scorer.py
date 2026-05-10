"""Goodness of Pronunciation (GOP) scoring across accent-marking phoneme
contrasts.

Standard GOP for a target phone p over aligned frames F is:

    GOP(p) = (1 / |F|) * sum_f [ log P(p | f) - max_q log P(q | f) ]

A score near 0 means the acoustic model is confident the speaker produced
the target phone; a large negative score means the frames looked more like
a competing phone. Contrasts (e.g. retroflex vs. alveolar /t/) are scored by
comparing GOP(target) against GOP(competing phone) on the same frames, which
surfaces exactly the substitutions associated with Indian English accents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.gop.aligner import PhonemeSegment


@dataclass(frozen=True)
class ContrastResult:
    name: str
    target_phone: str
    competing_phone: str
    gop_target: float
    gop_competing: float

    @property
    def contrast_score(self) -> float:
        """Positive = closer to target phone; negative = closer to competitor."""
        return self.gop_target - self.gop_competing


class AcousticPosteriorModel:
    """Interface for a frame-level phone posterior model, e.g. a CTC/hybrid
    acoustic model producing P(phone | frame) for a fixed phone inventory."""

    phone_inventory: list[str]

    def frame_posteriors(self, audio_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        """Returns an array of shape (num_frames, num_phones) of P(phone | frame)."""
        raise NotImplementedError


def gop_for_phone(posteriors: np.ndarray, phone_inventory: list[str], phone: str) -> float:
    if phone not in phone_inventory:
        raise ValueError(f"phone '{phone}' not in inventory")
    idx = phone_inventory.index(phone)
    log_posteriors = np.log(np.clip(posteriors, 1e-8, 1.0))
    target_log_prob = log_posteriors[:, idx]
    max_log_prob = log_posteriors.max(axis=1)
    return float(np.mean(target_log_prob - max_log_prob))


def score_contrasts(
    audio_path: str,
    segments: list[PhonemeSegment],
    model: AcousticPosteriorModel,
    contrasts: list[dict],
) -> list[ContrastResult]:
    """For each configured contrast, find segments matching the target phone
    and score both the target and the competing phone on those frames."""
    results = []
    for contrast in contrasts:
        phones = contrast["phones"]
        if len(phones) != 2:
            raise ValueError(
                f"contrast '{contrast.get('name', '?')}' must have exactly 2 phones, got {phones}"
            )
        target_phone, competing_phone = phones
        matching = [s for s in segments if s.phone == target_phone]
        if not matching:
            continue
        target_scores = []
        competing_scores = []
        for seg in matching:
            posteriors = model.frame_posteriors(audio_path, seg.start_sec, seg.end_sec)
            target_scores.append(gop_for_phone(posteriors, model.phone_inventory, target_phone))
            competing_scores.append(
                gop_for_phone(posteriors, model.phone_inventory, competing_phone)
            )
        results.append(
            ContrastResult(
                name=contrast["name"],
                target_phone=target_phone,
                competing_phone=competing_phone,
                gop_target=float(np.mean(target_scores)),
                gop_competing=float(np.mean(competing_scores)),
            )
        )
    return results
