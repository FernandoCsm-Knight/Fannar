"""MLP residual com a mesma estrutura de sondagem da ResNet do artigo.

`stem` + seis blocos residuais + cabeca linear, com os mesmos nomes de estado (`stem`,
`b1`...`b6`) e a mesma interface (`get_block`, `block_names`, `out_conv`). Assim as tres
fontes do artigo se aplicam sem nenhuma mudanca: o estado pos-ReLU do bloco e A, a derivada
da margem com respeito a ele e Γ, e o "filtro efetivo ativado" e p = Σ_c h[c] W[c,:] = Wᵀh,
com W a camada linear de saida do bloco -- a mesma formula da eq. `fonte_parametros`, com a
posicao espacial reduzida a um unico ponto.

O bloco e pos-ativado, h_ℓ = ReLU(F(h_{ℓ−1}) + h_{ℓ−1}), como na ResNet, para que os sete
estados formem uma cadeia unica e a datacao por intervencao tenha a mesma leitura.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Stem(nn.Module):
    def __init__(self, d_in: int, width: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, width, bias=False)
        self.bn = nn.BatchNorm1d(width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.fc(x)))

    @property
    def out_conv(self) -> nn.Linear:
        return self.fc


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.fc1(x)))
        return self.act(self.bn2(self.fc2(h)) + x)

    @property
    def out_conv(self) -> nn.Linear:
        """A camada cujas saidas sao o eixo de estado do bloco (analogo ao `conv2`)."""
        return self.fc2


class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, width: int = 256, n_blocks: int = 6) -> None:
        super().__init__()
        self.stem = Stem(d_in, width)
        self._names = ["stem"]
        for i in range(1, n_blocks + 1):
            setattr(self, f"b{i}", Block(width))
            self._names.append(f"b{i}")
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self._names:
            x = getattr(self, name)(x)
        return self.head(x)

    @property
    def block_names(self) -> list[str]:
        return list(self._names)

    def get_block(self, name: str) -> nn.Module:
        return getattr(self, name)
