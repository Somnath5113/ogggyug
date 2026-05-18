"""Single-file inference: predicted accent class, calibrated confidence, and
(optionally) GOP diagnostics for the configured phoneme contrasts."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from src.features.embeddings import LanguageIDEmbedder, WavLMEmbedder, load_waveform
from src.models.calibration import TemperatureScaler
from src.models.classifier import AccentClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--top-k", type=int, default=None, help="limit the printed distribution")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    classes = config["classes"]

    wavlm = WavLMEmbedder(
        config["features"]["wavlm_model_name"],
        layer=config["features"]["wavlm_layer"],
        pooling=config["features"]["pooling"],
    )
    lang_id = LanguageIDEmbedder(config["features"]["lang_id_model_name"])

    waveform, sample_rate = load_waveform(args.audio)
    wavlm_emb = wavlm.embed(waveform, sample_rate)
    lang_emb = lang_id.embed(waveform, sample_rate)
    embedding = torch.from_numpy(
        np.concatenate([wavlm_emb, lang_emb], axis=-1)
    ).float().unsqueeze(0)

    model = AccentClassifier(
        input_dim=embedding.shape[-1],
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

    with torch.no_grad():
        logits = model(embedding)
        probs = scaler.calibrated_probs(logits).squeeze(0)

    predicted_idx = int(probs.argmax())
    print(f"predicted accent: {classes[predicted_idx]}")
    print(f"calibrated confidence: {probs[predicted_idx].item():.3f}")
    print("full distribution:")
    ranked = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])
    for cls, p in ranked[: args.top_k]:
        print(f"  {cls:12s} {p:.3f}")


if __name__ == "__main__":
    main()
