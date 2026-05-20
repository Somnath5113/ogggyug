# Indian Accent Detection Model

A 7-class Indian-English accent classifier built on **frozen WavLM** speech
representations combined with **language-ID embeddings**. The model reaches
**70.5% top-1 accuracy** (chance level: 14.3%) on a held-out, speaker-disjoint
test set of **1,772 speakers**.

## Highlights

- **Frozen self-supervised backbone**: WavLM embeddings are extracted once
  and cached; only a lightweight classification head is trained on top,
  concatenated with language-ID embeddings for additional discriminative
  signal.
- **Speaker leakage fix**: an early version of the pipeline leaked speaker
  identity across the train/val/test split, inflating accuracy by more than
  25 points. The splitting logic now enforces strict speaker-disjoint
  partitions with a pinned, fixed set of held-out test speakers so results
  are reproducible and not an artifact of the split.
- **Confidence calibration**: raw softmax confidences were poorly calibrated.
  Temperature scaling (fit on a held-out calibration split) reduces the rate
  of high-confidence wrong predictions from 28% to 10%. The fitted
  temperature is clamped away from zero to avoid degenerate, pathologically
  sharp softmax outputs.
- **Phoneme-level pronunciation scoring**: a Goodness of Pronunciation (GOP)
  scorer evaluates 7 phoneme contrasts that are known markers of Indian
  English accents (e.g. retroflex vs. alveolar stops, /v/-/w/ merger).

## Pipeline

```
raw audio
   │
   ├─► WavLM (frozen) ──► pooled embedding ─┐
   │                                        ├─► concat ─► MLP head ─► 7-way softmax
   └─► Language-ID model (frozen) ──► embedding ─┘                       │
                                                                          ▼
                                                            temperature scaling (T)
                                                                          │
                                                                          ▼
                                                              calibrated confidence
```

A separate, independent path runs the **GOP scorer** on forced-aligned
phoneme boundaries to produce per-contrast pronunciation scores, used as
supplementary diagnostic output rather than as classifier input.

## Requirements

```
pip install -r requirements.txt
```

The language-ID embedder depends on `speechbrain`, which pulls in its own
model checkpoint on first use.

## Repository layout

```
configs/
  default.yaml          # paths, class list, training hyperparameters
src/
  data/
    manifest.py          # dataset manifest schema + loading
    splits.py            # speaker-disjoint train/val/test splitting
  features/
    embeddings.py         # WavLM + language-ID embedding extraction/caching
  models/
    classifier.py         # MLP head over concatenated embeddings
    calibration.py         # temperature scaling
  gop/
    aligner.py             # phoneme alignment interface
    scorer.py               # GOP computation over 7 contrasts
  train.py                 # training loop
  evaluate.py              # accuracy / calibration metrics on test split
  predict.py               # single-file inference (class + calibrated confidence + GOP)
scripts/
  extract_embeddings.py    # batch embedding extraction CLI
  build_splits.py          # CLI to (re)build speaker-disjoint splits
tests/
  test_splits.py           # regression test: no speaker overlap across splits
  test_calibration.py      # temperature scaling sanity checks
  test_gop.py               # GOP scorer sanity checks
```

## Results

| Metric | Value |
|---|---|
| Classes | 7 |
| Chance accuracy | 14.3% |
| Test speakers (held out, speaker-disjoint) | 1,772 |
| Top-1 accuracy | 70.5% |
| High-confidence wrong-prediction rate, pre-calibration | 28% |
| High-confidence wrong-prediction rate, post-calibration | 10% |

The pre-fix pipeline (speaker leakage across splits) reported accuracy more
than 25 points higher than the number above; that number is not reported
here because it does not reflect generalization to unseen speakers.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Build speaker-disjoint splits from a manifest of (audio_path, speaker_id, accent_label)
python scripts/build_splits.py --manifest data/manifest.csv --out configs/splits.json

# 2. Extract and cache frozen WavLM + language-ID embeddings
python scripts/extract_embeddings.py --manifest data/manifest.csv --out data/embeddings/

# 3. Train the classifier head + fit temperature scaling
python -m src.train --config configs/default.yaml

# 4. Evaluate on the held-out test split (optionally dump metrics as JSON)
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt --output-json metrics.json

# 5. Run inference (classification + calibrated confidence + GOP diagnostics) on one file
python -m src.predict --audio path/to/clip.wav --checkpoint checkpoints/best.pt --top-k 3
```

## Notes on methodology

- **Why speaker-disjoint splits matter**: accent classifiers trained on
  frozen speech embeddings can trivially memorize speaker identity rather
  than accent if the same speaker appears in both train and test. This
  collapses the task into speaker recognition and produces accuracy numbers
  that do not transfer to new speakers.
- **Why temperature scaling**: the classifier head, trained with standard
  cross-entropy, tends to produce overconfident softmax outputs. A single
  scalar temperature learned on a calibration split (held out from both
  training and test) rescales logits post-hoc without changing the
  argmax/accuracy, only the reliability of the reported confidence.
- **Why GOP on 7 contrasts**: raw classifier accuracy doesn't explain *why*
  an accent was predicted. The GOP scorer gives interpretable, phoneme-level
  evidence (e.g. which specific contrast most influenced the prediction) on
  top of the black-box classification.
