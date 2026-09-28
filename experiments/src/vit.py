"""Transformer de visao com a mesma estrutura de sondagem da ResNet (`src/model.py`).

Existe para uma pergunta so: o metodo depende da arquitetura? As tres fontes do artigo sao
definidas sobre um *estado* com um eixo de canais e um eixo de posicao, e sobre a camada cuja
saida define esse eixo de canais -- nada nelas e convolucional. Este modelo torna isso
verificavel: mesmos sete pontos de sondagem (`stem`, `b1`...`b6`), mesma interface
(`block_names`, `get_block`, `out_conv`, `head`), e nenhuma mudanca no codigo de analise.

Tres escolhas de implementacao servem a essa comparabilidade:

  estado espacial   cada bloco devolve (B, C, H, W), nao (B, N, C). A atencao e calculada
                    sobre os N = H·W tokens e o resultado volta a grade; assim o campo de
                    energia por posicao, os conjuntos de nivel e a Gram cruzada leem o estado
                    do transformer exatamente como leem o da ResNet.
  MLP em 1x1        as duas projecoes do bloco sao convolucoes 1x1, identicas a camadas
                    lineares por token. A segunda e o `out_conv`: a camada cujas saidas sao o
                    eixo de estado, de onde sai o filtro efetivo ativado p = Wᵀh e a metrica
                    de canais W Wᵀ -- o analogo exato do `conv2` do bloco residual.
  tokenizador conv  em lugar de uma projecao de patches, tres convolucoes 3x3 com passo 2
                    (CCT, Hassani et al., 2021). Com 6,6 mil imagens de treino, a projecao de
                    patches pura nao sai do lugar; o tokenizador convolucional e o que torna
                    o treino do zero viavel nesta escala.

A normalizacao final do transformer fica *dentro* do ultimo bloco, e nao entre o bloco e a
cabeca, para que o ponto de sondagem continue sendo "o tensor que o proximo modulo le" em
todos os sete estados, como na ResNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm sobre o eixo de canais de um tensor (B, C, H, W)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Tokenizer(nn.Module):
    """Tokenizador convolucional: 128x128 -> 16x16 tokens, com posicao aprendida."""

    def __init__(self, dim: int, grid: int, c_in: int = 3) -> None:
        super().__init__()
        mid = dim // 2
        self.conv1 = nn.Conv2d(c_in, mid, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, dim, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.pos = nn.Parameter(torch.zeros(1, dim, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        return h + self.pos

    @property
    def out_conv(self) -> nn.Conv2d:
        return self.conv3

    @property
    def ablatable(self) -> bool:
        return False  # unico caminho dos pixels para a rede


class Attention(nn.Module):
    """Atencao multi-cabeca sobre os H·W tokens, entrando e saindo em (B, C, H, W)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-2, -1) for t in qkv.unbind(1))  # (B, heads, N, dim/heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(-2, -1).reshape(b, c, h, w))


class DropPath(nn.Module):
    """Caminho estocastico: no treino, o ramo residual e zerado em parte das amostras."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class Block(nn.Module):
    """Bloco pre-norm: x + atencao, depois x + MLP de duas convolucoes 1x1."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0, final_norm: bool = False,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path)
        hidden = int(dim * mlp_ratio)
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.final = LayerNorm2d(dim) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return self.final(x)

    @property
    def out_conv(self) -> nn.Conv2d:
        """A camada cujas saidas sao o eixo de estado do bloco (analoga ao `conv2`)."""
        return self.fc2

    @property
    def ablatable(self) -> bool:
        return True  # todo bloco e residual e preserva a forma


class PetsViT(nn.Module):
    """tokenizador -> 6 blocos de atencao -> pooling global -> linear."""

    def __init__(self, n_classes: int = 37, dim: int = 192, depth: int = 6, heads: int = 3,
                 size: int = 128, drop_path: float = 0.1) -> None:
        super().__init__()
        grid = size // 8
        self.stem = Tokenizer(dim, grid)
        self._names = ["stem"]
        for i in range(1, depth + 1):
            # a taxa cresce com a profundidade, como em Huang et al. (2016)
            setattr(self, f"b{i}", Block(dim, heads, final_norm=(i == depth),
                                         drop_path=drop_path * i / depth))
            self._names.append(f"b{i}")
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, n_classes))
        self.head_width = dim
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self._names:
            x = getattr(self, name)(x)
        return self.head(x)

    @property
    def block_names(self) -> list[str]:
        return list(self._names)

    def get_block(self, name: str) -> nn.Module:
        return getattr(self, name)

    def ablatable_blocks(self) -> list[str]:
        return [n for n in self._names if self.get_block(n).ablatable]
