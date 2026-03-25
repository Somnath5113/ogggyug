import torch

from src.models.calibration import TemperatureScaler, high_confidence_error_rate


def test_temperature_scaling_preserves_argmax():
    torch.manual_seed(0)
    logits = torch.randn(200, 7) * 3
    labels = logits.argmax(dim=-1)

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    raw_preds = logits.argmax(dim=-1)
    scaled_preds = scaler(logits).argmax(dim=-1)
    assert torch.equal(raw_preds, scaled_preds)


def test_high_confidence_error_rate_basic():
    probs = torch.tensor(
        [
            [0.95, 0.05],
            [0.05, 0.95],
            [0.6, 0.4],
        ]
    )
    labels = torch.tensor([1, 1, 0])  # first prediction (class 0) is wrong and confident

    rate = high_confidence_error_rate(probs, labels, threshold=0.8)
    assert rate == 1.0  # the only high-confidence prediction (row 0) is wrong


def test_high_confidence_error_rate_no_confident_predictions():
    probs = torch.tensor([[0.5, 0.5], [0.55, 0.45]])
    labels = torch.tensor([0, 1])
    assert high_confidence_error_rate(probs, labels, threshold=0.9) == 0.0


def test_temperature_is_clamped_away_from_zero():
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.log_temperature.fill_(-50.0)  # would otherwise underflow to ~0
    assert scaler.temperature.item() >= 1e-3
