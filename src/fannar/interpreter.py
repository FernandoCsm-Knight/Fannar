"""High-level interface: a trained model, the objects to interpret, and everything else.

>>> import fannar as fa
>>> interp = fa.Interpreter(model, X_test, y_test, layers=["stem", "b1", "b2"],
...                         reference_model=untrained_copy)          # floor (optional)
>>> report = interp.evaluate()                                        # readings per layer
>>> report.to_frame(); report.to_markdown(); interp.plot_report(report)
>>> interp.plot_embedding(); interp.plot_gram("b2"); interp.plot_spectrum("b2")
>>> expl = interp.explain_pair(0)           # nearest object with another prediction
>>> expl.top(5); expl.plot()
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .pairs import PairExplanation, curve_area, nearest_unlike, pair_scores, swap_curve
from .pipeline import KernelPipeline, get_pipeline, positions_pipeline
from .readings import evaluate as evaluate_space
from .report import Report
from .sources import FunctionSources, Sources, TreeSources
from .space import GramSpace


def _is_torch(model) -> bool:
    try:
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model, nn.Module)


def _is_sklearn_tree(model) -> bool:
    return hasattr(model, "tree_") and hasattr(model, "decision_path")


class Interpreter:
    """Builds the representation space of a trained model over chosen objects, per layer.

    Parameters
    ----------
    model:
        A trained PyTorch classifier, a fitted scikit-learn decision tree, or anything else when
        ``extractor`` is given.
    X:
        The objects to interpret (array, tensor or, for PyTorch, a DataLoader).
    y:
        True labels of ``X`` (needed for the correctness, confusion and containment readings).
    layers:
        Layers to probe (PyTorch module names); ignored for trees and custom extractors.
    pipeline:
        A :class:`KernelPipeline`, or a preset name (``"paper"`` by default). For your own Gram
        construction, pass a pipeline whose components use your kernels, or call
        :meth:`space_from_grams`.
    reference_model:
        Optional model of the same architecture used as the *floor* (typically the same network
        with its initial weights). Its geometry is measured against the trained model's behaviour.
    baselines:
        Other pipelines or preset names evaluated alongside (default: activations alone).
    extractor:
        ``callable(X) -> Sources`` (or :class:`FunctionSources`) for models that are neither
        PyTorch nor sklearn trees.
    **source_options:
        Passed to :class:`~fannar.sources.TorchSources` (``target``, ``fixed_class``, ``grid``,
        ``weight_modules``, ``device``, ``batch_size``...).
    """

    def __init__(self, model, X, y=None, *, layers: Sequence[str] | None = None,
                 pipeline: KernelPipeline | str = "paper", reference_model=None,
                 baselines: Sequence[KernelPipeline | str] | dict = ("activations",),
                 extractor: Callable | None = None, feature_names: Sequence[str] | None = None,
                 class_names: Sequence[str] | None = None, k: int = 5, **source_options) -> None:
        self.model, self.X, self.k = model, X, k
        self.y = None if y is None else np.asarray(y)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.class_names = list(class_names) if class_names is not None else None
        self.extractor = self._make_extractor(model, extractor, layers, source_options)
        self.reference_extractor = (None if reference_model is None
                                    else self._make_extractor(reference_model, None, layers, source_options))
        self.pipeline = get_pipeline(pipeline)
        if _is_sklearn_tree(model) and pipeline == "paper":
            self.pipeline = KernelPipeline(["path", "slack", "classes"])
            if tuple(baselines) == ("activations",):
                baselines = {"path": KernelPipeline(["path"])}
        if isinstance(baselines, dict):
            self.baselines = {k: get_pipeline(v) for k, v in baselines.items()}
        else:
            self.baselines = {(b if isinstance(b, str) else f"baseline{t}"): get_pipeline(b)
                              for t, b in enumerate(baselines)}
        self.sources: Sources | None = None
        self.reference_sources: Sources | None = None
        self.spaces: dict[str, GramSpace] = {}

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _make_extractor(model, extractor, layers, options):
        if extractor is not None:
            known = (FunctionSources, TreeSources)
            try:
                from .sources.torch import TorchSources
                known = known + (TorchSources,)
            except ImportError:  # pragma: no cover
                pass
            return extractor if isinstance(extractor, known) else FunctionSources(extractor)
        if _is_torch(model):
            from .sources.torch import TorchSources
            return TorchSources(model, layers, **options)
        if _is_sklearn_tree(model):
            return TreeSources(model, **{k: v for k, v in options.items() if k in ("scale", "depths")})
        raise TypeError("model is neither a torch.nn.Module nor a sklearn tree; pass `extractor=`")

    def fit(self) -> "Interpreter":
        """Extracts the sources and builds the representation space of every layer."""
        self.sources = self.extractor(self.X)
        if self.reference_extractor is not None:
            self.reference_sources = self.reference_extractor(self.X)
        self.spaces = {layer: self._build(self.pipeline, self.sources, layer, "joint") for layer in self.layers}
        return self

    def _ensure(self) -> None:
        if self.sources is None:
            self.fit()

    def _build(self, pipeline: KernelPipeline, sources: Sources, layer: str, name: str) -> GramSpace:
        available = sources.layers[layer]
        missing = [c for c in pipeline.names if c not in available]
        if missing:
            raise KeyError(f"layer {layer!r} has no source(s) {missing}; available: {list(available)}")
        return pipeline(sources=available, labels=self.y, predictions=self.sources.predictions if self.sources else None,
                        name=f"{name} · {layer}")

    @property
    def layers(self) -> list[str]:
        if self.sources is None:
            self.fit()
        return self.sources.layer_names

    @property
    def predictions(self) -> np.ndarray:
        self._ensure()
        return self.sources.predictions

    def space(self, layer: str) -> GramSpace:
        """The :class:`GramSpace` of one layer."""
        self._ensure()
        return self.spaces[layer]

    def baseline_space(self, name: str, layer: str) -> GramSpace:
        self._ensure()
        return self._build(self.baselines[name], self.sources, layer, name)

    def floor_space(self, layer: str) -> GramSpace:
        """The same pipeline on the reference (e.g. untrained) model."""
        self._ensure()
        if self.reference_sources is None:
            raise ValueError("no reference_model was given")
        return self._build(self.pipeline, self.reference_sources, layer, "floor")

    def space_from_grams(self, grams: dict[str, np.ndarray], pipeline: KernelPipeline | None = None,
                         name: str = "custom") -> GramSpace:
        """A space from Gram matrices you built yourself (``{component: (n, n)}``), composed by
        ``pipeline`` (default: the interpreter's). Components not in ``grams`` must be sources."""
        self._ensure()
        pipe = pipeline or KernelPipeline(list(grams))
        return pipe(grams=grams, labels=self.y, predictions=self.predictions, name=name)

    # ------------------------------------------------------------------ readings
    def evaluate(self, layers: Sequence[str] | None = None, include_baselines: bool = True,
                 include_floor: bool = True) -> Report:
        """Behaviour readings per layer for the joint space, the baselines and the floor.

        The floor uses the reference model's *geometry* against the trained model's *behaviour*
        (the "cross" floor), so the only difference between a reading and its floor is training.
        """
        self._ensure()
        if self.y is None:
            raise ValueError("evaluate() needs the true labels `y`")
        layers = list(layers or self.layers)
        preds, probs = self.sources.predictions, self.sources.probabilities
        run = lambda sp: evaluate_space(sp, self.y, preds, probs, self.k)  # noqa: E731
        values = {"joint": {layer: run(self.spaces[layer]) for layer in layers}}
        if include_baselines:
            for name in self.baselines:
                ok = [layer for layer in layers if self._has(self.baselines[name], layer)]
                if ok:
                    values[name] = {layer: run(self.baseline_space(name, layer)) for layer in ok}
        if include_floor and self.reference_sources is not None:
            values["floor"] = {layer: run(self.floor_space(layer)) for layer in layers}
        accuracy = float((preds == self.y).mean())
        return Report(layers, values, {"accuracy": accuracy, "n": len(self.y), "pipeline": repr(self.pipeline)})

    # ------------------------------------------------------------------ pairs
    def explain_pair(self, i: int, j: int | None = None, layers: Sequence[str] | None = None) -> PairExplanation:
        """Which features distinguish object ``i`` from ``j`` for the model (``j`` defaults to the
        nearest object with a different prediction). Vector inputs only."""
        self._ensure()
        X = self._matrix()
        j = nearest_unlike(X, self.predictions, i) if j is None else j
        scores = pair_scores(self.extractor, self.pipeline, X[i], X[j], layers)
        return PairExplanation(i, j, X[i], X[j], scores, int(self.predictions[i]), int(self.predictions[j]),
                               self.feature_names)

    def swap_test(self, pairs: int | Sequence[tuple[int, int]] = 50, layer: str | None = None,
                  predict: Callable | None = None, seed: int = 0) -> dict:
        """Swap-test area per ranking (method, |Delta x|, random among differing features),
        averaged over pairs. ``pairs`` is a number of random correctly-predicted objects (each with
        its nearest unlike neighbour) or explicit ``(i, j)`` pairs."""
        self._ensure()
        X = self._matrix()
        predict = predict or self._predict_fn()
        rng = np.random.default_rng(seed)
        if isinstance(pairs, int):
            pool = np.flatnonzero(self.predictions == self.y) if self.y is not None else np.arange(len(X))
            chosen = rng.choice(pool, size=min(pairs, len(pool)), replace=False)
            pairs = [(int(i), nearest_unlike(X, self.predictions, int(i))) for i in chosen]
        layer = layer or self.layers[-1]
        areas = {"method": [], "absdiff": [], "random": []}
        for i, j in pairs:
            s = pair_scores(self.extractor, self.pipeline, X[i], X[j], [layer])[layer]
            differ = np.flatnonzero(np.abs(X[j] - X[i]) > 1e-12)
            rest = np.setdiff1d(np.arange(X.shape[1]), differ)
            orders = {"method": np.argsort(-s, kind="stable"), "absdiff": np.argsort(-np.abs(X[j] - X[i]), kind="stable"),
                      "random": np.concatenate([rng.permutation(differ), rest])}
            for key, order in orders.items():
                areas[key].append(curve_area(swap_curve(predict, X[i], X[j], int(self.predictions[j]), order)))
        return {k: float(np.mean(v)) for k, v in areas.items()} | {"pairs": len(pairs), "layer": layer}

    # ------------------------------------------------------------------ positions of one input
    def energy_map(self, x, layer: str, max_side: int = 32) -> np.ndarray:
        """Retained spectral energy of every position of one input (e.g. an image), as a map.

        Positions are the objects; components use linear kernels without the angular transform
        (which would give every position the same energy)."""
        from .sources.torch import TorchSources
        if not isinstance(self.extractor, TorchSources):
            raise TypeError("energy maps need a PyTorch model")
        src = self.extractor.positions(x, layer, max_side)
        names = [s for s in ("parameters", "activations", "gradients") if s in src]
        space = positions_pipeline(names)(sources=src)
        shape = self.extractor.position_shape(x, layer, max_side)
        return space.energies()["E_par"].reshape(shape)

    # ------------------------------------------------------------------ plots
    def plot_report(self, report: Report | None = None, readings=None):
        from .plots import plot_report
        return plot_report(report or self.evaluate(), readings)

    def plot_embedding(self, layers: Sequence[str] | None = None, color_by: str = "prediction",
                       include_floor: bool = True, include_baselines: bool = True):
        from .plots import plot_embedding
        self._ensure()
        layers = list(layers or self.layers)
        rows = {"joint": [self.spaces[layer] for layer in layers]}
        if include_baselines:
            for name in self.baselines:
                if all(self._has(self.baselines[name], layer) for layer in layers):
                    rows[name] = [self.baseline_space(name, layer) for layer in layers]
        if include_floor and self.reference_sources is not None:
            rows["floor"] = [self.floor_space(layer) for layer in layers]
        colors = self.predictions if color_by == "prediction" else self.y
        return plot_embedding(rows, layers, colors, self.y, self.predictions, self.class_names)

    def plot_gram(self, layer: str):
        from .plots import plot_gram
        return plot_gram(self.space(layer), self.y if self.y is not None else self.predictions)

    def plot_spectrum(self, layer: str):
        from .plots import plot_spectrum
        return plot_spectrum(self.space(layer))

    def plot_energy_map(self, x, layer: str, max_side: int = 32, image=None):
        from .plots import plot_energy_map
        return plot_energy_map(self.energy_map(x, layer, max_side), image)

    # ------------------------------------------------------------------ helpers
    def _has(self, pipeline: KernelPipeline, layer: str) -> bool:
        return all(c in self.sources.layers[layer] for c in pipeline.names)

    def _matrix(self) -> np.ndarray:
        X = self.X
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("pair explanations need vector inputs (n, F)")
        return X

    def _predict_fn(self) -> Callable:
        if _is_sklearn_tree(self.model):
            return self.model.predict
        if _is_torch(self.model):
            import torch
            model = self.model
            device = next(model.parameters()).device

            def predict(rows):
                model.eval()
                with torch.no_grad():
                    return model(torch.as_tensor(rows, device=device)).argmax(1).cpu().numpy()
            return predict
        raise TypeError("pass `predict=` for this model")
