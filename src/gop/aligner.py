"""Phoneme forced-alignment interface.

Wraps an external forced aligner (e.g. Montreal Forced Aligner / an ASR
acoustic model) behind a small interface so the GOP scorer does not need to
know which aligner produced the phoneme boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonemeSegment:
    phone: str
    start_sec: float
    end_sec: float


class ForcedAligner:
    """Interface for a forced aligner. Implementations must produce phoneme
    segments with timestamps for a given audio file and its transcript."""

    def align(self, audio_path: str, transcript: str) -> list[PhonemeSegment]:
        raise NotImplementedError(
            "Plug in a forced aligner (e.g. Montreal Forced Aligner) here."
        )
