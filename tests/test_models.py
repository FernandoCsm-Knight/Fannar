"""Testes de integração com PyTorch."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fannar.models import FannarExtractor
from fannar.pipelines import default_layer_pipeline


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_hooks_removed_after_extraction():
    model = TinyNet().eval()
    extractor = FannarExtractor(model, ["conv"], capture=["activations"])
    x = torch.randn(2, 3, 8, 8)
    extractor.extract(x)
    # nenhum hook deve restar no módulo
    assert len(model.conv._forward_hooks) == 0


def test_extract_weights_activations_gradients():
    model = TinyNet().eval()
    extractor = FannarExtractor(model, ["conv"], capture=["weights", "activations", "gradients"])
    x = torch.randn(2, 3, 8, 8)
    reps = extractor.extract(x, target=0)
    r = reps["conv"]
    assert r.weights is not None and r.weights.shape[0] == 6
    assert r.activations is not None and r.activations.shape[0] == 6
    assert r.gradients is not None and r.gradients.shape[0] == 6


def test_full_pipeline_cpu():
    model = TinyNet().eval()
    extractor = FannarExtractor(model, ["conv"], capture=["weights", "activations", "gradients"])
    x = torch.randn(2, 3, 8, 8)
    reps = extractor.extract(x, target=0)
    pipe = default_layer_pipeline()
    geo = pipe.analyze(reps["conv"], tau=0.9, compute_profile=True)
    assert geo.K_total.shape == (6, 6)
    assert geo.coords_2d.shape == (6, 2)
    assert geo.profile is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_full_pipeline_gpu():
    model = TinyNet().cuda().eval()
    extractor = FannarExtractor(model, ["conv"], capture=["weights", "activations"])
    x = torch.randn(2, 3, 8, 8, device="cuda")
    reps = extractor.extract(x, target=0)
    assert reps["conv"].activations.is_cuda
