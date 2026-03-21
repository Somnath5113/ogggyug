"""Trains the accent classifier head on cached frozen embeddings, then fits
temperature scaling on the calibration split."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.manifest import load_manifest
from src.data.splits import load_splits, utterances_for_split
from src.features.embeddings import load_cached_embedding
from src.models.calibration import TemperatureScaler
from src.models.classifier import AccentClassifier


class EmbeddingDataset(Dataset):
    def __init__(self, utterances, embeddings_dir, class_to_idx):
        self.utterances = utterances
        self.embeddings_dir = embeddings_dir
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx):
        u = self.utterances[idx]
        key = os.path.splitext(os.path.basename(u.audio_path))[0]
        npz_path = os.path.join(self.embeddings_dir, f"{key}.npz")
        embedding = load_cached_embedding(npz_path)
        label = self.class_to_idx[u.accent_label]
        return torch.from_numpy(embedding).float(), label


def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(embeddings)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += len(labels)
    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = config["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    utterances = load_manifest(config["data"]["manifest"])
    splits = load_splits(config["data"]["splits_file"])
    embeddings_dir = config["data"]["embeddings_dir"]

    train_set = EmbeddingDataset(
        utterances_for_split(utterances, splits["train"]), embeddings_dir, class_to_idx
    )
    val_set = EmbeddingDataset(
        utterances_for_split(utterances, splits["val"]), embeddings_dir, class_to_idx
    )
    calib_set = EmbeddingDataset(
        utterances_for_split(utterances, splits["calibration"]), embeddings_dir, class_to_idx
    )

    sample_embedding, _ = train_set[0]
    input_dim = sample_embedding.shape[0]

    model = AccentClassifier(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
    )
    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"])

    os.makedirs(config["train"]["checkpoint_dir"], exist_ok=True)
    best_val_acc, patience_left = 0.0, config["train"]["early_stopping_patience"]

    for epoch in range(config["train"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, device=device)
        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = config["train"]["early_stopping_patience"]
            torch.save(
                model.state_dict(),
                os.path.join(config["train"]["checkpoint_dir"], "best.pt"),
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(config["train"]["checkpoint_dir"], "best.pt")))
    model.eval()

    calib_loader = DataLoader(calib_set, batch_size=len(calib_set))
    embeddings, labels = next(iter(calib_loader))
    with torch.no_grad():
        logits = model(embeddings.to(device)).cpu()

    scaler = TemperatureScaler()
    nll = scaler.fit(logits, labels, lr=config["calibration"]["lr"], max_iter=config["calibration"]["max_iter"])
    print(f"fitted temperature: {scaler.temperature.item():.4f} (calibration NLL {nll:.4f})")
    torch.save(
        scaler.state_dict(), os.path.join(config["train"]["checkpoint_dir"], "temperature.pt")
    )


if __name__ == "__main__":
    main()
