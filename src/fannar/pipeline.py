"""The kernel-transformation pipeline: component Grams, Hadamard composition, final transform.

::

    G = T( T_1(K_1) o T_2(K_2) o ... o T_d(K_d) )

where ``K_l`` is the raw Gram of source ``l`` (from a kernel, or supplied directly by the
user), ``T_l`` its admissible component transform, ``o`` the Hadamard product and ``T`` the
final admissible transform. By Schur's product theorem ``G`` is PSD for any choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .kernels import Kernel, Linear, as_kernel
from .space import GramSpace
from .transforms import Transform, angular, center, discarded_mass, hadamard_power, identity, psd_project, shift, trace_normalize


@dataclass
class Component:
    """One source of information: its kernel and its component transform."""

    kernel: Kernel | str = field(default_factory=Linear)
    transform: Transform = field(default_factory=lambda: angular() >> shift(1.0))

    def __post_init__(self) -> None:
        self.kernel = as_kernel(self.kernel)


class KernelPipeline:
    """Composes several sources of the same objects into one :class:`GramSpace`.

    Parameters
    ----------
    components:
        ``{source_name: Component}``, ``{source_name: kernel}`` or a list of source names
        (each gets the default component: linear kernel, angular transform, shift by ``J``).
    final:
        Transform applied to the Hadamard product (default ``center >> trace_normalize``).
    root:
        If True, use the ``d``-th Hadamard root of the product (then projected on the PSD cone);
        the discarded mass is recorded in ``space.info["root_discarded"]``.

    Examples
    --------
    >>> pipe = KernelPipeline(["activations", "gradients"])
    >>> space = pipe(sources={"activations": A, "gradients": Gr})           # from features
    >>> space = pipe(grams={"activations": KA, "gradients": KG})            # from your own Grams
    """

    def __init__(self, components: Mapping[str, Component | Kernel | str] | Sequence[str],
                 final: Transform | None = None, root: bool = False) -> None:
        if isinstance(components, Mapping):
            comps = {k: v if isinstance(v, Component) else Component(kernel=v) for k, v in components.items()}
        else:
            comps = {k: Component() for k in components}
        if not comps:
            raise ValueError("at least one component is required")
        self.components: dict[str, Component] = comps
        self.final = final if final is not None else center >> trace_normalize
        self.root = root

    @property
    def names(self) -> list[str]:
        return list(self.components)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}: {c.kernel.name} | {c.transform.name}" for k, c in self.components.items())
        return f"KernelPipeline({parts}; final={self.final.name}{'; root' if self.root else ''})"

    def component_grams(self, sources: Mapping | None = None, grams: Mapping | None = None) -> dict[str, np.ndarray]:
        """``T_l(K_l)`` for every component. A raw Gram in ``grams`` takes precedence over the
        kernel applied to ``sources``, so any component can be replaced by the user's own matrix."""
        sources, grams = sources or {}, grams or {}
        out = {}
        for name, comp in self.components.items():
            if name in grams:
                raw = np.asarray(grams[name], dtype=np.float64)
            elif name in sources:
                raw = comp.kernel(sources[name])
            else:
                raise KeyError(f"no source or Gram given for component {name!r}")
            out[name] = comp.transform(raw)
        return out

    def __call__(self, sources: Mapping | None = None, grams: Mapping | None = None, *, labels=None,
                 predictions=None, name: str = "", keep_components: bool = True) -> GramSpace:
        comps = self.component_grams(sources, grams)
        joint = None
        for g in comps.values():
            joint = g.copy() if joint is None else joint * g
        info = {"pipeline": repr(self)}
        if self.root and len(comps) > 1:
            powered = hadamard_power(1.0 / len(comps))(joint)
            info["root_discarded"] = discarded_mass(powered)
            joint = psd_project()(powered)
        G = self.final(joint)
        return GramSpace(G, labels, predictions, comps if keep_components else None, name, info)


# ---------------------------------------------------------------------- presets
PAPER_SOURCES = ("parameters", "activations", "gradients")


def paper_pipeline(sources: Sequence[str] = PAPER_SOURCES, shift_by: float = 1.0) -> KernelPipeline:
    """The instantiation of the paper: linear kernel, ``(T_a(K) + J) / 2`` per source, Hadamard
    product and ``T_tr o T_c``."""
    comp = lambda: Component(Linear(), angular() >> shift(shift_by))  # noqa: E731
    return KernelPipeline({s: comp() for s in sources}, final=center >> trace_normalize)


def single_source(source: str = "activations", angular_shift: bool = True) -> KernelPipeline:
    """One source alone, with the same component transform as the paper (for ablations), or the
    plain linear centred Gram (``angular_shift=False``), whose distances are the Euclidean
    distances between the features -- the geometry the linear CKA and RSA look at."""
    t = angular() >> shift(1.0) if angular_shift else identity
    return KernelPipeline({source: Component(Linear(), t)}, final=center >> trace_normalize)


def positions_pipeline(sources: Sequence[str] = PAPER_SOURCES) -> KernelPipeline:
    """For positions of one input as objects: linear kernels, no angular transform (it would fix
    every position's energy), Hadamard product and ``T_tr o T_c``."""
    return KernelPipeline({s: Component(Linear(), identity) for s in sources}, final=center >> trace_normalize)


PRESETS = {"paper": paper_pipeline, "activations": lambda: single_source("activations"),
           "gradients": lambda: single_source("gradients"), "parameters": lambda: single_source("parameters"),
           "euclidean": lambda: single_source("activations", angular_shift=False)}


def get_pipeline(obj) -> KernelPipeline:
    """A :class:`KernelPipeline` or the name of a preset (``"paper"``, ``"activations"``,
    ``"gradients"``, ``"parameters"``, ``"euclidean"``)."""
    if isinstance(obj, KernelPipeline):
        return obj
    if isinstance(obj, str):
        try:
            return PRESETS[obj]()
        except KeyError as exc:
            raise ValueError(f"unknown preset {obj!r}; choose from {sorted(PRESETS)}") from exc
    raise TypeError(f"cannot interpret {obj!r} as a pipeline")
