from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "encoder"))

from cevr.model import CEVRModel


def test_encoder_shapes_and_causality() -> None:
    ends = [25, 50, 75, 100, 125, 150, 200, 250]
    model = CEVRModel(channels=17, embed_dim=512, window_ends=ends)
    model.eval()
    torch.manual_seed(7)
    eeg = torch.randn(2, 17, 250)
    with torch.no_grad():
        baseline = model(eeg)["window_embeddings"]
    assert baseline.shape == (2, len(ends), 512)
    for idx, end in enumerate(ends[:-1]):
        perturbed = eeg.clone()
        perturbed[:, :, end:] += 10.0 * torch.randn_like(perturbed[:, :, end:])
        with torch.no_grad():
            changed = model(perturbed)["window_embeddings"]
        assert torch.allclose(baseline[:, : idx + 1], changed[:, : idx + 1], atol=1e-6)

