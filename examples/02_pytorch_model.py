"""Exemplo: modelo PyTorch simples, extração e análise de uma camada."""

from __future__ import annotations

import torch
import torch.nn as nn

import fannar as f
from fannar.models import FannarExtractor
from fannar.pipelines import default_layer_pipeline


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main() -> None:
    torch.manual_seed(0)
    model = SmallCNN().eval()
    x = torch.randn(4, 3, 16, 16)

    extractor = FannarExtractor(
        model, layers=["conv2"], capture=["weights", "activations", "gradients"]
    )
    reps = extractor.extract(inputs=x, target=0)

    geo = default_layer_pipeline().analyze(reps["conv2"], tau=0.9, compute_profile=True)
    print("K_total:", tuple(geo.K_total.shape))
    print("cobertura:", geo.coverage.detail)
    print("perfil:", geo.profile.as_dict())


if __name__ == "__main__":
    main()
