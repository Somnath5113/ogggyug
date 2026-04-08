"""Speaker-disjoint train/val/calibration/test splitting.

The classifier operates on frozen speech embeddings, which makes it easy for
a model to key off speaker identity rather than accent if the same speaker's
utterances land in more than one split. This module guarantees that every
speaker_id is assigned to exactly one split, and that a fixed, pinned set of
test speakers can be reused across runs so reported numbers are comparable.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from src.data.manifest import Utterance


def assign_speakers_to_splits(
    utterances: list[Utterance],
    split_ratios: dict[str, float],
    seed: int = 42,
    pinned_test_speakers: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return {split_name: [speaker_id, ...]} with no speaker in more than one split."""
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")
    negative = {name: r for name, r in split_ratios.items() if r < 0}
    if negative:
        raise ValueError(f"split_ratios must be non-negative, got {negative}")

    speakers = sorted({u.speaker_id for u in utterances})
    pinned_test_speakers = pinned_test_speakers or set()
    unknown_pinned = pinned_test_speakers - set(speakers)
    if unknown_pinned:
        raise ValueError(f"pinned test speakers not present in manifest: {unknown_pinned}")

    remaining = [s for s in speakers if s not in pinned_test_speakers]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    result: dict[str, list[str]] = defaultdict(list)
    result["test"].extend(sorted(pinned_test_speakers))

    n_remaining = len(remaining)
    test_already = len(pinned_test_speakers)
    total_speakers = len(speakers)
    target_test_total = round(split_ratios.get("test", 0.0) * total_speakers)
    extra_test_needed = max(0, target_test_total - test_already)

    cursor = 0
    result["test"].extend(remaining[cursor: cursor + extra_test_needed])
    cursor += extra_test_needed

    other_splits = [name for name in split_ratios if name != "test"]
    other_ratio_sum = sum(split_ratios[name] for name in other_splits)
    n_left = n_remaining - extra_test_needed

    consumed = 0
    for i, name in enumerate(other_splits):
        if i == len(other_splits) - 1:
            count = n_left - consumed
        else:
            count = round((split_ratios[name] / other_ratio_sum) * n_left)
        result[name].extend(remaining[cursor: cursor + count])
        cursor += count
        consumed += count

    _assert_disjoint(result)
    return dict(result)


def _assert_disjoint(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, speaker_ids in splits.items():
        for speaker_id in speaker_ids:
            if speaker_id in seen:
                raise AssertionError(
                    f"speaker {speaker_id} appears in both '{seen[speaker_id]}' and "
                    f"'{split_name}' splits"
                )
            seen[speaker_id] = split_name


def utterances_for_split(
    utterances: list[Utterance], speaker_ids: list[str]
) -> list[Utterance]:
    speaker_set = set(speaker_ids)
    return [u for u in utterances if u.speaker_id in speaker_set]


def save_splits(splits: dict[str, list[str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)


def load_splits(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
