"""Token sequences: a PyTorch transformer (or any model) whose probed outputs are ``(N, L, C)``.

>>> from fannar.text import TextInterpreter, wrap_hf
>>> model = wrap_hf(hf_model, pad_token_id=tokenizer.pad_token_id)   # Hugging Face -> logits
>>> interp = TextInterpreter(model, input_ids, y)                    # layers: the encoder blocks
>>> interp.evaluate()

Hidden states are read with the channels on the last axis (``layout="tokens"``) and pooled over
the sequence to ``grid`` positions (``grid=8`` by default). With ``layers=None`` the probed layers
are the blocks of the model's longest ``nn.ModuleList`` (``encoder.layer.*``, ``transformer.h.*``,
``layers.*``...), see :func:`block_names`.

``transformers`` is not a dependency: :func:`wrap_hf` relies only on the call
``model(input_ids=..., attention_mask=...)`` returning an object with ``.logits``, a tuple or a tensor.
"""

from __future__ import annotations

from typing import Sequence

from .interpreter import Interpreter, require_torch

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is optional for the rest of the library
    torch = nn = None


def _need_torch() -> None:
    require_torch("fannar.text")


def _logits(out):
    if isinstance(out, torch.Tensor):
        return out
    logits = getattr(out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot read logits from a model output of type {type(out).__name__}")


if nn is not None:
    class HFClassifier(nn.Module):
        """A classifier taking one tensor of token ids and returning its logits ``(N, K)``.

        The wrapped model is stored as ``self.model``, so the probed layer names gain the prefix
        ``model.`` (``model.bert.encoder.layer.0``...). With ``pad_token_id`` the attention mask is
        ``input_ids != pad_token_id``; otherwise every token is attended to.
        """

        def __init__(self, model: nn.Module, pad_token_id: int | None = None, **forward_kwargs) -> None:
            super().__init__()
            self.model = model
            self.pad_token_id = pad_token_id
            self.forward_kwargs = forward_kwargs

        def forward(self, input_ids):
            kwargs = dict(self.forward_kwargs)
            if self.pad_token_id is not None:
                kwargs["attention_mask"] = (input_ids != self.pad_token_id).long()
            return _logits(self.model(input_ids=input_ids, **kwargs))
else:  # pragma: no cover
    HFClassifier = None


def wrap_hf(model, pad_token_id: int | None = None, **forward_kwargs):
    """Wraps a Hugging Face sequence classifier (or anything called as ``model(input_ids=...)``)
    into an :class:`HFClassifier` whose ``forward(input_ids)`` returns the logits.

    Extra keyword arguments are passed to every call of the wrapped model.
    """
    _need_torch()
    return HFClassifier(model, pad_token_id, **forward_kwargs)


def block_names(model) -> list[str]:
    """Names of the children of the longest ``nn.ModuleList`` in ``model``: the transformer blocks.

    Raises ``ValueError`` when the model has no ``nn.ModuleList``; pass ``layers=`` explicitly then.
    """
    _need_torch()
    lists = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m)]
    if not lists:
        raise ValueError("no nn.ModuleList found in the model; pass `layers=` explicitly")
    name, blocks = max(lists, key=lambda item: len(item[1]))  # first of the longest, in module order
    return [f"{name}.{t}" if name else str(t) for t in range(len(blocks))]


class TextInterpreter(Interpreter):
    """:class:`~fannar.Interpreter` for PyTorch models over token sequences.

    Parameters
    ----------
    model:
        A trained PyTorch classifier taking one tensor (e.g. token ids ``(N, L)``) and returning
        logits ``(N, K)``. Wrap a Hugging Face model with :func:`wrap_hf` first.
    X:
        The sequences: an array or tensor of token ids (or embeddings), or a ``DataLoader``.
    y:
        True labels (integers ``0 .. K-1``).
    layers:
        Module names to probe; ``None`` uses :func:`block_names`.
    grid:
        Positions kept along the sequence when pooling each hidden state (``None`` keeps every token,
        which requires sequences of the same length).
    **kwargs:
        Passed to :class:`~fannar.Interpreter` (``reference_model``, ``pipeline``, ``baselines``,
        ``class_names``, ``k``) and from it to :class:`~fannar.TorchSources` (``target``,
        ``fixed_class``, ``device``, ``batch_size``, ``weight_modules``...).
    """

    def __init__(self, model, X, y=None, *, layers: Sequence[str] | None = None, grid: int | None = 8,
                 **kwargs) -> None:
        _need_torch()
        if layers is None:
            layers = block_names(model)
        kwargs.setdefault("layout", "tokens")
        super().__init__(model, X, y, layers=layers, grid=grid, **kwargs)


__all__ = ["TextInterpreter", "HFClassifier", "wrap_hf", "block_names"]
