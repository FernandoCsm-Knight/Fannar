"""CIFAR-10 ResNet with named residual blocks as the probe points.

Six basic blocks in three stages of widths 64 / 128 / 256, plus the stem, giving
seven states along the depth. Every probe point is the *residual stream after the
block's final ReLU*, so hooking a block gives one unambiguous tensor -- the thing
the next block reads -- and there is no ambiguity about where inside a block the
state lives.

The architecture is chosen for what it makes measurable, not only for accuracy:

  identity ablation   a block can be replaced by the identity exactly when its
                      shortcut already is one, i.e. when it neither changes width
                      nor strides. Here that is b1, b2, b4 and b6 -- four of six,
                      against two of six for a plain VGG stack, and unlike there
                      the deletion is absorbed by the residual path instead of
                      handing the next block activations it has never seen.

  logit lens          the trained head reads 256 channels, so it can be applied
                      to any state of stage 3 with no parameter fitted. Stages 1
                      and 2 are 64 and 128 wide and stay out of reach; the width
                      growth is the price of a conventional backbone.

  channel metric      each block exposes `out_conv`, the convolution whose output
                      channels define the state. Its filter Gram is the
                      similarity between that layer's parameters, and it is
                      square in the state's own channel axis -- which is what
                      lets it act as a metric on that axis rather than being
                      bolted on as a third pseudo-source.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Stem(nn.Module):
    """3x3 conv into the first stage width.

    `stride=1` keeps the input resolution (32x32 inputs); `stride=2` halves it, the usual
    choice for larger images (128x128 here), which keeps the shallow stages' feature maps small
    enough to train on a 6 GB GPU.
    """

    def __init__(self, c_in: int = 3, c_out: int = 64, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    @property
    def out_conv(self) -> nn.Conv2d:
        return self.conv

    @property
    def ablatable(self) -> bool:
        return False  # the stem is the only path from pixels into the network


class BasicBlock(nn.Module):
    """Standard two-conv residual block, post-activation."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=False)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))

    @property
    def out_conv(self) -> nn.Conv2d:
        """The convolution whose output channels are the block's state axis."""
        return self.conv2

    @property
    def ablatable(self) -> bool:
        """True when dropping the block leaves a well-formed chain."""
        return isinstance(self.shortcut, nn.Identity)


class CIFAR10ResNet(nn.Module):
    """stem -> 3 stages x 2 blocks (64/128/256) -> GAP -> linear."""

    def __init__(
        self,
        n_classes: int = 10,
        widths: tuple[int, ...] = (64, 128, 256),
        blocks_per_stage: int = 2,
        stem_width: int = 64,
        stem_stride: int = 1,
    ) -> None:
        super().__init__()
        self.stem = Stem(3, stem_width, stem_stride)
        self._names = ["stem"]
        c_in = stem_width
        index = 0
        for stage, width in enumerate(widths):
            for b in range(blocks_per_stage):
                # stride 2 at every stage boundary except the first
                stride = 2 if (b == 0 and stage > 0) else 1
                index += 1
                name = f"b{index}"
                setattr(self, name, BasicBlock(c_in, width, stride))
                self._names.append(name)
                c_in = width
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, n_classes)
        )
        self.head_width = c_in

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

    def lens_blocks(self) -> list[str]:
        """States the trained head can read directly, with nothing fitted."""
        return [
            n for n in self._names
            if self.get_block(n).out_conv.out_channels == self.head_width
        ]

    def width(self, name: str) -> int:
        return self.get_block(name).out_conv.out_channels
