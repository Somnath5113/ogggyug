"""Frozen WavLM + language-ID embedding extraction.

Both backbones are used purely as feature extractors: no gradients flow into
them, and their outputs are cached to disk so the classifier head can be
trained/iterated on quickly without re-running the backbones.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torchaudio


class WavLMEmbedder:
    def __init__(self, model_name: str, layer: int = -1, pooling: str = "mean", device: str = "cpu"):
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        if pooling not in ("mean", "max"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.device = device
        self.layer = layer
        self.pooling = pooling
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        if sample_rate != self.extractor.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.extractor.sampling_rate
            )
        inputs = self.extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.extractor.sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer]  # (1, T, H)
        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden.max(dim=1).values
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")
        return pooled.squeeze(0).cpu().numpy()


class LanguageIDEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        from speechbrain.inference.classifiers import EncoderClassifier

        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=model_name, run_opts={"device": device}
        )

    def embed(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        with torch.no_grad():
            emb = self.model.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def extract_and_cache(
    audio_path: str,
    out_dir: str,
    wavlm: WavLMEmbedder,
    lang_id: LanguageIDEmbedder,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    key = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(out_dir, f"{key}.npz")
    if os.path.exists(out_path):
        return out_path

    waveform, sample_rate = load_waveform(audio_path)
    wavlm_emb = wavlm.embed(waveform, sample_rate)
    lang_emb = lang_id.embed(waveform, sample_rate)
    np.savez(out_path, wavlm=wavlm_emb, lang_id=lang_emb)
    return out_path


def load_cached_embedding(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    return np.concatenate([data["wavlm"], data["lang_id"]], axis=-1)
