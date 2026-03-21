"""CLI to batch-extract and cache frozen WavLM + language-ID embeddings for
every utterance in a manifest."""
from __future__ import annotations

import argparse

from tqdm import tqdm

from src.data.manifest import load_manifest
from src.features.embeddings import LanguageIDEmbedder, WavLMEmbedder, extract_and_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    utterances = load_manifest(args.manifest)
    wavlm = WavLMEmbedder(
        config["features"]["wavlm_model_name"],
        layer=config["features"]["wavlm_layer"],
        pooling=config["features"]["pooling"],
    )
    lang_id = LanguageIDEmbedder(config["features"]["lang_id_model_name"])

    for utterance in tqdm(utterances, desc="extracting embeddings"):
        extract_and_cache(utterance.audio_path, args.out, wavlm, lang_id)


if __name__ == "__main__":
    main()
