"""Evaluates the trained classifier on the held-out, speaker-disjoint test
split and reports both raw and calibrated metrics."""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.manifest import load_manifest
from src.data.splits import load_splits, utterances_for_split
from src.models.calibration import TemperatureScaler, high_confidence_error_rate
from src.models.classifier import AccentClassifier
from src.train import EmbeddingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default=None, help="optional path to dump metrics as JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    classes = config["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    utterances = load_manifest(config["data"]["manifest"])
    splits = load_splits(config["data"]["splits_file"])
    test_set = EmbeddingDataset(
        utterances_for_split(utterances, splits["test"]),
        config["data"]["embeddings_dir"],
        class_to_idx,
    )

    sample_embedding, _ = test_set[0]
    model = AccentClassifier(
        input_dim=sample_embedding.shape[0],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    scaler = TemperatureScaler()
    temp_path = os.path.join(config["train"]["checkpoint_dir"], "temperature.pt")
    if os.path.exists(temp_path):
        scaler.load_state_dict(torch.load(temp_path, map_location="cpu"))

    loader = DataLoader(test_set, batch_size=len(test_set))
    embeddings, labels = next(iter(loader))
    with torch.no_grad():
        logits = model(embeddings)
        raw_probs = F.softmax(logits, dim=-1)
        calibrated_probs = scaler.calibrated_probs(logits)

    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    raw_hce = high_confidence_error_rate(raw_probs, labels)
    calibrated_hce = high_confidence_error_rate(calibrated_probs, labels)

    print(f"test speakers: {len(splits['test'])}")
    print(f"test utterances: {len(test_set)}")
    print(f"top-1 accuracy: {accuracy * 100:.1f}%")
    print(f"high-confidence wrong-prediction rate (raw): {raw_hce * 100:.1f}%")
    print(f"high-confidence wrong-prediction rate (calibrated): {calibrated_hce * 100:.1f}%")
    print(f"fitted temperature: {scaler.temperature.item():.3f}")

    if args.output_json:
        metrics = {
            "test_speakers": len(splits["test"]),
            "test_utterances": len(test_set),
            "top1_accuracy": accuracy,
            "high_confidence_error_rate_raw": raw_hce,
            "high_confidence_error_rate_calibrated": calibrated_hce,
            "temperature": scaler.temperature.item(),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
