"""CLI to build (or rebuild) speaker-disjoint train/val/calibration/test splits."""
from __future__ import annotations

import argparse
import json

from src.data.manifest import load_manifest
from src.data.splits import assign_speakers_to_splits, save_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pinned-test-speakers",
        help="path to a JSON file containing a list of speaker_ids to force into the test split",
        default=None,
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()

    utterances = load_manifest(args.manifest)

    pinned = set()
    if args.pinned_test_speakers:
        with open(args.pinned_test_speakers) as f:
            pinned = set(json.load(f))

    split_ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "calibration": args.calibration_ratio,
        "test": args.test_ratio,
    }
    splits = assign_speakers_to_splits(
        utterances, split_ratios, seed=args.seed, pinned_test_speakers=pinned
    )
    save_splits(splits, args.out)

    for name, speaker_ids in splits.items():
        print(f"{name}: {len(speaker_ids)} speakers")


if __name__ == "__main__":
    main()
