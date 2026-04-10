import csv

import pytest

from src.data.manifest import load_manifest


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "speaker_id", "accent_label"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_manifest_parses_rows(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [{"audio_path": "a.wav", "speaker_id": "spk0", "accent_label": "hindi"}],
    )
    utterances = load_manifest(str(path))
    assert len(utterances) == 1
    assert utterances[0].accent_label == "hindi"


def test_load_manifest_rejects_empty_field(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [{"audio_path": "", "speaker_id": "spk0", "accent_label": "hindi"}],
    )
    with pytest.raises(ValueError):
        load_manifest(str(path))
