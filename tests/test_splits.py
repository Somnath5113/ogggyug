"""Regression test guarding against the speaker-leakage bug: no speaker_id
may appear in more than one split."""
from src.data.manifest import Utterance
from src.data.splits import assign_speakers_to_splits


def _make_utterances(num_speakers: int, utterances_per_speaker: int) -> list[Utterance]:
    utterances = []
    for s in range(num_speakers):
        for u in range(utterances_per_speaker):
            utterances.append(
                Utterance(
                    audio_path=f"spk{s}_utt{u}.wav",
                    speaker_id=f"spk{s}",
                    accent_label="hindi" if s % 2 == 0 else "tamil",
                )
            )
    return utterances


def test_splits_are_speaker_disjoint():
    utterances = _make_utterances(num_speakers=200, utterances_per_speaker=5)
    ratios = {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    splits = assign_speakers_to_splits(utterances, ratios, seed=0)

    all_ids = [sid for ids in splits.values() for sid in ids]
    assert len(all_ids) == len(set(all_ids)), "a speaker leaked across splits"


def test_pinned_test_speakers_are_respected():
    utterances = _make_utterances(num_speakers=100, utterances_per_speaker=3)
    ratios = {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    pinned = {"spk0", "spk1", "spk2"}
    splits = assign_speakers_to_splits(utterances, ratios, seed=1, pinned_test_speakers=pinned)

    assert pinned.issubset(set(splits["test"]))


def test_split_sizes_roughly_match_ratios():
    utterances = _make_utterances(num_speakers=1000, utterances_per_speaker=2)
    ratios = {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    splits = assign_speakers_to_splits(utterances, ratios, seed=2)

    total = sum(len(ids) for ids in splits.values())
    assert total == 1000
    assert 0.65 <= len(splits["train"]) / total <= 0.75
