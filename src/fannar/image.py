"""Images: a PyTorch CNN or vision transformer whose probed outputs are ``(N, C, H, W)``.

>>> from fannar.image import ImageInterpreter
>>> interp = ImageInterpreter(model, loader, y_test, layers=["layer1", "layer2", "layer3"],
...                           reference_model=untrained_copy, class_names=classes)
>>> interp.evaluate()
>>> interp.plot_energy_map(image, "layer2", image=rgb)   # where the energy sits inside one image

Each probed output is average-pooled to ``grid x grid`` positions (``grid=4`` by default), so an
object carries ``C x grid**2`` values per source. The objects can be an array, a tensor or a
``DataLoader`` (the first element of every batch is used). With images as objects there are no
per-feature explanations; :meth:`energy_map` reads the positions of one image instead.
"""

from __future__ import annotations

from typing import Sequence

from .interpreter import Interpreter, require_torch


class ImageInterpreter(Interpreter):
    """:class:`~fannar.Interpreter` for PyTorch image classifiers.

    Parameters
    ----------
    model:
        A trained PyTorch classifier returning logits ``(N, K)``.
    X:
        Images ``(n, C, H, W)`` as an array or tensor, or a ``DataLoader``.
    y:
        True labels (integers ``0 .. K-1``).
    layers:
        Module names to probe, as in ``model.named_modules()``.
    grid:
        Positions kept per spatial axis when pooling each probed output (``None`` keeps all of them,
        which makes the sources much larger).
    class_names:
        Names of the classes, used by the plots and the error explanations.
    **kwargs:
        Passed to :class:`~fannar.Interpreter` (``reference_model``, ``pipeline``, ``baselines``, ``k``)
        and from it to :class:`~fannar.TorchSources` (``target``, ``fixed_class``, ``device``,
        ``batch_size``...).
    """

    def __init__(self, model, X, y=None, *, layers: Sequence[str] | None = None, grid: int | None = 4,
                 class_names: Sequence[str] | None = None, **kwargs) -> None:
        require_torch("fannar.image")
        kwargs.setdefault("layout", "channels")
        super().__init__(model, X, y, layers=layers, class_names=class_names, grid=grid, **kwargs)


__all__ = ["ImageInterpreter"]
