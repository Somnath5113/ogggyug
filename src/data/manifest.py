"""Dataset manifest loading.

A manifest is a CSV with columns: audio_path, speaker_id, accent_label.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    audio_path: str
    speaker_id: str
    accent_label: str


def load_manifest(path: str) -> list[Utterance]:
    utterances = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"audio_path", "speaker_id", "accent_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")
        for row in reader:
            utterances.append(
                Utterance(
                    audio_path=row["audio_path"],
                    speaker_id=row["speaker_id"],
                    accent_label=row["accent_label"],
                )
            )
    return utterances
