"""Exemplo mínimo: construção do espaço de tipos e energia residual.

Roda sem modelo: usa tensores sintéticos para pesos/ativações/gradientes.
"""

from __future__ import annotations

import torch

import fannar as f


def main() -> None:
    torch.manual_seed(0)
    C, d = 32, 64  # 32 canais, 64 features por fonte

    W = torch.randn(C, d)
    A = torch.randn(C, d)
    G = torch.randn(C, d)

    # Pipelines por fonte
    from fannar.pipelines import LayerRepresentations, default_layer_pipeline

    reps = LayerRepresentations(weights=W, activations=A, gradients=G)
    geo = default_layer_pipeline().analyze(
        reps, tau=0.95, gradient_direction=torch.rand(C)
    )

    print("Cobertura espectral:", geo.coverage.detail)
    print("Energia residual do gradiente:",
          None if geo.residual is None else float(geo.residual.value))
    print("Perfil explicativo:", geo.profile.as_dict())
    print("Coords 2D shape:", tuple(geo.coords_2d.shape))


if __name__ == "__main__":
    main()
