import numpy as np

from src.gop.aligner import PhonemeSegment
from src.gop.scorer import AcousticPosteriorModel, gop_for_phone, score_contrasts


class FakeAcousticModel(AcousticPosteriorModel):
    """Always confident about the true target phone: gop should be ~0."""

    phone_inventory = ["t", "T", "d", "D", "v", "w"]

    def frame_posteriors(self, audio_path, start_sec, end_sec):
        num_frames = 10
        posteriors = np.full((num_frames, len(self.phone_inventory)), 0.01)
        posteriors[:, 0] = 0.95  # confidently "t"
        return posteriors


def test_gop_for_phone_confident_prediction_near_zero():
    model = FakeAcousticModel()
    posteriors = model.frame_posteriors("dummy.wav", 0, 1)
    score = gop_for_phone(posteriors, model.phone_inventory, "t")
    assert score == 0.0  # target phone has the max posterior on every frame


def test_gop_for_phone_penalizes_wrong_phone():
    model = FakeAcousticModel()
    posteriors = model.frame_posteriors("dummy.wav", 0, 1)
    score = gop_for_phone(posteriors, model.phone_inventory, "T")
    assert score < 0  # retroflex T never wins here, so GOP should be negative


def test_score_contrasts_produces_result_per_matching_segment():
    model = FakeAcousticModel()
    segments = [PhonemeSegment(phone="t", start_sec=0.0, end_sec=0.1)]
    contrasts = [{"name": "retroflex_vs_alveolar_t", "phones": ["t", "T"]}]

    results = score_contrasts("dummy.wav", segments, model, contrasts)

    assert len(results) == 1
    assert results[0].name == "retroflex_vs_alveolar_t"
    assert results[0].contrast_score > 0  # model favors the alveolar target
