"""Sources of a PyTorch model: activations (A), contrastive gradient (Gamma), activated parameters (W).

For each probed module ("layer") and each object:

* ``activations`` -- the module output, pooled to a common grid of positions;
* ``gradients``   -- derivative of the contrastive margin
  ``m = f_c - mean_{c' != c} f_{c'}`` with respect to that output, on the same grid;
* ``parameters``  -- the filters the object activated, ``p[:, s] = sum_c A[c, s] W[c, :]``, with
  ``W`` the weight of the module that produces the output (the last layer with a weight whose
  first dimension equals the number of output channels).

Output layouts: ``(N, C)`` -> one position; ``(N, C, H, W)`` -> adaptive average pooling to
``grid x grid``; ``(N, L, C)`` (tokens) -> pooling over ``L`` to ``grid`` positions.

The reference class ``c`` of the margin matters. With ``target="predicted"`` the gradient
carries the model's own decision and readings about the decision become circular; use
``target="fixed"`` (a fixed class for every object) for readings that are tested against the
model's predictions.
"""

from __future__ import annotations

import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Sources

SOURCE_NAMES = ("parameters", "activations", "gradients")


def _tensor_output(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot probe a module whose output is {type(out).__name__}")


def _pool(t: torch.Tensor, grid: int | None, layout: str, cap: bool = False) -> torch.Tensor:
    """``(N, ...)`` -> ``(N, C, P)``. With ``cap`` the grid is only an upper bound (never upsamples)."""
    if t.dim() == 2:
        return t[:, :, None]
    if t.dim() == 4:
        if grid and not (cap and max(t.shape[2:]) <= grid):
            t = F.adaptive_avg_pool2d(t, (min(grid, t.shape[2]) if cap else grid, min(grid, t.shape[3]) if cap else grid))
        return t.flatten(2)
    if t.dim() == 3:
        if layout == "tokens":  # (N, L, C)
            t = t.transpose(1, 2)
        if grid and not (cap and t.shape[2] <= grid):
            t = F.adaptive_avg_pool1d(t, grid)
        return t
    return t.flatten(1)[:, :, None]


def _batches(X, batch_size: int):
    if isinstance(X, torch.utils.data.DataLoader):
        for b in X:
            yield b[0] if isinstance(b, (tuple, list)) else b
        return
    X = torch.as_tensor(np.asarray(X) if not isinstance(X, torch.Tensor) else X)
    for s in range(0, len(X), batch_size):
        yield X[s : s + batch_size]


class TorchSources:
    """Extracts the three sources of the paper from any ``torch.nn.Module`` classifier.

    Parameters
    ----------
    model:
        A trained classifier returning logits ``(N, K)``.
    layers:
        Module names (as in ``model.named_modules()``) or modules to probe. ``None`` probes every
        direct child that has parameters.
    target:
        ``"fixed"`` (class ``fixed_class`` for every object), ``"predicted"`` or an int.
    weight_modules:
        Optional ``{layer: module or name}`` giving the layer whose weight defines the parameter
        source; by default it is found automatically.
    sources:
        Which sources to extract (subset of ``("parameters", "activations", "gradients")``).
    grid:
        Positions kept per spatial/token axis (``None`` keeps all).
    margin:
        ``"contrastive"`` (default) or a callable ``fn(logits, target) -> (N,)``.
    """

    def __init__(self, model: nn.Module, layers: Sequence[str | nn.Module] | None = None, *,
                 target: str | int = "fixed", fixed_class: int = 0,
                 weight_modules: Mapping[str, str | nn.Module] | None = None,
                 sources: Sequence[str] = SOURCE_NAMES, grid: int | None = 4, layout: str = "channels",
                 margin: str | Callable = "contrastive", device=None, batch_size: int = 256,
                 dtype=torch.float32) -> None:
        self.model = model
        self.device = torch.device(device) if device else next(model.parameters()).device
        named = dict(model.named_modules())
        if layers is None:
            layers = [n for n, m in model.named_children() if any(True for _ in m.parameters())]
        self.layers: dict[str, nn.Module] = {}
        for layer in layers:
            if isinstance(layer, str):
                if layer not in named:
                    raise KeyError(f"module {layer!r} not found; available: {list(named)[:20]}...")
                self.layers[layer] = named[layer]
            else:
                name = next((n for n, m in named.items() if m is layer), None) or f"layer{len(self.layers)}"
                self.layers[name] = layer
        self.target, self.fixed_class = target, fixed_class
        self.weight_modules = {k: (named[v] if isinstance(v, str) else v) for k, v in (weight_modules or {}).items()}
        unknown = set(sources) - set(SOURCE_NAMES)
        if unknown:
            raise ValueError(f"unknown sources {unknown}; choose from {SOURCE_NAMES}")
        self.sources, self.grid, self.layout = tuple(sources), grid, layout
        self.margin, self.batch_size, self.dtype = margin, batch_size, dtype
        self._cap = False

    # ------------------------------------------------------------------ helpers
    def _weight_for(self, name: str, module: nn.Module, channels: int) -> torch.Tensor | None:
        def fits(m):
            w = getattr(m, "weight", None)
            return isinstance(w, torch.Tensor) and w.dim() >= 2 and w.shape[0] == channels

        if name in self.weight_modules:
            w = self.weight_modules[name].weight
        elif fits(module):
            w = module.weight
        else:
            # the last module, up to and including the probed one, whose weight produces `channels`
            # outputs: conv2 inside a residual block, or the Linear before a probed activation
            ordered = list(self.model.modules())
            inside = set(map(id, module.modules()))
            stop = max(t for t, m in enumerate(ordered) if id(m) in inside)
            w = next((m.weight for m in reversed(ordered[: stop + 1]) if fits(m)), None)
            if w is None:
                return None
        w = w.detach().reshape(w.shape[0], -1)
        if w.shape[0] != channels:
            raise ValueError(f"weight of {name!r} has {w.shape[0]} rows but the output has {channels} channels")
        return w

    def _targets(self, logits: torch.Tensor) -> torch.Tensor:
        n = len(logits)
        if self.target == "predicted":
            return logits.argmax(1)
        c = self.fixed_class if self.target == "fixed" else int(self.target)
        return torch.full((n,), c, dtype=torch.long, device=logits.device)

    def _margin(self, logits: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        if callable(self.margin):
            return self.margin(logits, cls)
        rows = torch.arange(len(logits), device=logits.device)
        own = logits[rows, cls]
        return own - (logits.sum(1) - own) / (logits.shape[1] - 1)

    # ------------------------------------------------------------------ extraction
    def __call__(self, X) -> Sources:
        store: dict[str, torch.Tensor] = {}
        handles = []

        def hook(name):
            def fn(_m, _i, out):
                t = _tensor_output(out)
                if t.requires_grad:
                    t.retain_grad()
                store[name] = t
            return fn

        for name, m in self.layers.items():
            handles.append(m.register_forward_hook(hook(name)))
        chunks = {name: {s: [] for s in self.sources} for name in self.layers}
        weights: dict[str, torch.Tensor | None] = {}
        logits_all = []
        was_training = self.model.training
        self.model.eval()
        try:
            for xb in _batches(X, self.batch_size):
                xb = xb.to(self.device, self.dtype) if xb.is_floating_point() else xb.to(self.device)
                store.clear()
                self.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = self.model(xb)
                    need_grad = "gradients" in self.sources
                    if need_grad:
                        self._margin(logits, self._targets(logits)).sum().backward()
                for name in self.layers:
                    out = store[name]
                    a = _pool(out.detach(), self.grid, self.layout, self._cap).double()
                    if "activations" in self.sources:
                        chunks[name]["activations"].append(a.cpu())
                    if "gradients" in self.sources:
                        g = out.grad if out.grad is not None else torch.zeros_like(out)
                        chunks[name]["gradients"].append(_pool(g.detach(), self.grid, self.layout, self._cap).double().cpu())
                    if "parameters" in self.sources:
                        if name not in weights:
                            weights[name] = self._weight_for(name, self.layers[name], a.shape[1])
                            if weights[name] is None:
                                warnings.warn(f"no weight found for layer {name!r}; parameter source skipped")
                        w = weights[name]
                        if w is not None:
                            p = torch.einsum("ncp,cf->nfp", a, w.double().to(a.device))
                            chunks[name]["parameters"].append(p.cpu())
                logits_all.append(logits.detach().double().cpu())
        finally:
            for h in handles:
                h.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        layers = {name: {s: torch.cat(v).numpy() for s, v in per.items() if v} for name, per in chunks.items()}
        logits = torch.cat(logits_all).numpy()
        z = np.exp(logits - logits.max(1, keepdims=True))
        return Sources(layers, logits.argmax(1), z / z.sum(1, keepdims=True), logits,
                       {"target": self.target, "fixed_class": self.fixed_class, "grid": self.grid})

    # ------------------------------------------------------------------ positions of one input
    def positions(self, x, layer: str, max_side: int | None = 32) -> dict[str, np.ndarray]:
        """Sources with the *positions* of one input as objects: ``{source: (P, C or F)}``.

        Used for energy maps over an image; ``max_side`` caps the grid (the Gram is ``P x P``).
        """
        x = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)[None]
        grid, self.grid, self._cap = self.grid, max_side, True
        try:
            src = self(x)
        finally:
            self.grid, self._cap = grid, False
        return {s: v[0].T for s, v in src.layers[layer].items()}

    def position_shape(self, x, layer: str, max_side: int | None = 32) -> tuple[int, ...]:
        """Spatial shape of the position grid returned by :meth:`positions` for ``layer``."""
        shape = {}
        def grab(_m, _i, out):  # must return None: a returned value would replace the module output
            shape.setdefault("s", _tensor_output(out).shape)

        h = self.layers[layer].register_forward_hook(grab)
        try:
            with torch.no_grad():
                self.model.eval()
                xb = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)[None].to(self.device, self.dtype)
                self.model(xb)
        finally:
            h.remove()
        s = shape["s"]
        if len(s) == 4:
            return tuple(min(max_side, v) if max_side else v for v in s[2:])
        if len(s) == 3:
            length = s[1] if self.layout == "tokens" else s[2]
            return (min(max_side, length) if max_side else length,)
        return (1,)
