"""The representation space induced by a PSD Gram matrix: geometry and spectral energy.

A PSD matrix ``G`` admits vectors ``phi_i`` with ``G_ij = <phi_i, phi_j>``; everything here is
computed from ``G`` alone, never from the ``phi_i``:

* geometry -- the pseudometric ``d(i, j)^2 = G_ii + G_jj - 2 G_ij``, neighbourhoods, level sets
  and representational equivalence;
* spectrum -- ``G = V Lambda V^T``, the retained subspace ``V_r``, the per-object mode energy
  ``E_ik = lambda_k V_ik^2`` and the orthogonal-energy ratio ``E_ratio``;
* comparison -- alignment with another Gram (CKA) and the decomposition
  ``G_hat = a K_hat + R`` into the part another Gram explains and the residual.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .transforms import discarded_mass


class GramSpace:
    """Representation space of ``n`` objects given by a symmetric PSD Gram matrix.

    Parameters
    ----------
    G:
        ``(n, n)`` symmetric positive semidefinite matrix (any admissible transformation of a
        kernel Gram, or a matrix the user built).
    labels, predictions:
        Optional per-object integers, stored for the readings and plots.
    components:
        Optional ``{name: (n, n)}`` component Grams that produced ``G``, kept for inspection.
    """

    def __init__(self, G, labels=None, predictions=None, components: dict | None = None,
                 name: str = "", info: dict | None = None) -> None:
        G = np.asarray(G, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(f"G must be square, got shape {G.shape}")
        self.G = 0.5 * (G + G.T)
        self.labels = None if labels is None else np.asarray(labels)
        self.predictions = None if predictions is None else np.asarray(predictions)
        self.components = components or {}
        self.name = name
        self.info = info or {}

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.G)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"GramSpace(n={self.n}{', ' + self.name if self.name else ''})"

    def is_psd(self, tol: float = 1e-6) -> bool:
        """True if the smallest eigenvalue is >= ``-tol * trace``."""
        return bool(self.eigenvalues_raw[0] >= -tol * max(abs(np.trace(self.G)), 1e-300))

    def subset(self, idx) -> "GramSpace":
        """The sub-space of a subset of objects (the submatrix, not re-transformed)."""
        idx = np.asarray(idx)
        take = lambda a: None if a is None else a[idx]  # noqa: E731
        return GramSpace(self.G[np.ix_(idx, idx)], take(self.labels), take(self.predictions),
                         {k: v[np.ix_(idx, idx)] for k, v in self.components.items()}, self.name, self.info)

    # ------------------------------------------------------------------ geometry
    @cached_property
    def distances(self) -> np.ndarray:
        """``(n, n)`` pseudometric ``d(i, j) = sqrt(G_ii + G_jj - 2 G_ij)``. O(n^2)."""
        d = np.diag(self.G)
        d2 = np.maximum(d[:, None] + d[None, :] - 2.0 * self.G, 0.0)
        np.fill_diagonal(d2, 0.0)
        return np.sqrt(d2)

    def distance(self, i: int, j: int) -> float:
        return float(np.sqrt(max(self.G[i, i] + self.G[j, j] - 2 * self.G[i, j], 0.0)))

    def distance_function(self, i0: int) -> np.ndarray:
        """``p_{i0}(j) = d(i0, j)`` for every object ``j``."""
        return self.distances[i0].copy()

    def neighbors(self, i0: int, k: int) -> np.ndarray:
        """The ``k`` objects closest to ``i0`` (excluding itself), nearest first. O(n)."""
        d = self.distances[i0].copy()
        d[i0] = np.inf
        k = min(k, self.n - 1)
        part = np.argpartition(d, k - 1)[:k]
        return part[np.argsort(d[part], kind="stable")]

    def level_band(self, i0: int, r: float, delta: float) -> np.ndarray:
        """``Gamma_{r, delta}(i0) = {j : |d(i0, j) - r| <= delta}``."""
        return np.flatnonzero(np.abs(self.distances[i0] - r) <= delta)

    def level_set(self, i0: int, r: float, tol: float = 1e-9) -> np.ndarray:
        """``Gamma_r(i0) = {j : d(i0, j) = r}`` up to ``tol``."""
        return self.level_band(i0, r, tol)

    def equivalence_classes(self, tol: float = 1e-9) -> list[np.ndarray]:
        """Groups of objects at pseudodistance ``<= tol`` (indistinguishable for this space)."""
        seen = np.zeros(self.n, dtype=bool)
        out = []
        for i in range(self.n):
            if seen[i]:
                continue
            group = np.flatnonzero((self.distances[i] <= tol) & ~seen)
            seen[group] = True
            out.append(group)
        return out

    # ------------------------------------------------------------------ spectrum
    @cached_property
    def eigenvalues_raw(self) -> np.ndarray:
        """Ascending eigenvalues without clipping (negative ones reveal a non-PSD input)."""
        return np.linalg.eigvalsh(self.G)

    @cached_property
    def _eig(self):
        vals, vecs = np.linalg.eigh(self.G)
        order = np.argsort(vals)[::-1]
        return np.maximum(vals[order], 0.0), vecs[:, order]

    @property
    def eigenvalues(self) -> np.ndarray:
        """Descending eigenvalues, clipped at zero. O(n^3), computed once."""
        return self._eig[0]

    @property
    def eigenvectors(self) -> np.ndarray:
        """Columns ordered as :attr:`eigenvalues`."""
        return self._eig[1]

    def participation_rank(self) -> int:
        """``round((sum lambda)^2 / sum lambda^2)`` from the trace and the Frobenius norm, O(n^2)."""
        tr = float(np.trace(self.G))
        fro2 = float(np.sum(self.G * self.G))
        return max(1, int(round(tr * tr / fro2))) if fro2 > 0 else 1

    def rank(self, rule: str | int = "participation", tau: float = 0.95) -> int:
        """Number of retained modes: ``"participation"``, ``"tau"`` (cumulative energy >= tau)
        or an explicit integer."""
        if isinstance(rule, (int, np.integer)):
            return int(max(1, min(rule, self.n)))
        if rule == "participation":
            return self.participation_rank()
        if rule == "tau":
            vals = self.eigenvalues
            return int(np.searchsorted(np.cumsum(vals) / max(vals.sum(), 1e-300), tau) + 1)
        raise ValueError(f"unknown rank rule {rule!r}")

    def mode_energy(self) -> np.ndarray:
        """``(n, n)`` matrix ``E_ik = lambda_k V_ik^2``; each row sums to ``G_ii``."""
        return self.eigenvalues[None, :] * self.eigenvectors**2

    def energies(self, r: str | int = "participation", tau: float = 0.95) -> dict:
        """Retained, orthogonal and total energy per object, and ``E_ratio = E_perp / G_ii``."""
        k = self.rank(r, tau)
        mode = self.mode_energy()
        total = mode.sum(1)
        par, perp = mode[:, :k].sum(1), mode[:, k:].sum(1)
        return {"rank": k, "E_par": par, "E_perp": perp, "E_total": total,
                "E_ratio": perp / np.maximum(total, 1e-300),
                "retained_fraction": float(self.eigenvalues[:k].sum() / max(self.eigenvalues.sum(), 1e-300))}

    def embedding(self, dim: int = 2) -> tuple[np.ndarray, float]:
        """Coordinates ``sqrt(lambda_k) V_ik`` of the ``dim`` dominant modes (kernel PCA of G)
        and the fraction of the trace they retain."""
        vals, vecs = self.eigenvalues[:dim], self.eigenvectors[:, :dim]
        return vecs * np.sqrt(vals), float(vals.sum() / max(self.eigenvalues.sum(), 1e-300))

    def containment(self, labels=None, r: str | int | None = None) -> float:
        """Fraction of the centred class-indicator directions inside ``V_r``.

        ``r`` defaults to ``K - 1`` (number of classes minus one), which makes the value
        comparable between spaces; the participation rank of each space is not.
        """
        labels = self._labels(labels)
        classes = np.unique(labels)
        y = (labels[:, None] == classes[None, :]).astype(float)
        y -= y.mean(0, keepdims=True)
        y /= np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
        k = self.rank(len(classes) - 1 if r is None else r)
        proj = self.eigenvectors[:, :k].T @ y
        return float((proj**2).sum() / max((y**2).sum(), 1e-12))

    # ------------------------------------------------------------------ comparison
    def cka(self, other) -> float:
        """Centred kernel alignment with another Gram (``GramSpace`` or array)."""
        a, b = _centered_unit(self.G), _centered_unit(_as_array(other))
        return float((a * b).sum())

    def decompose(self, reference) -> dict:
        """``G_hat = a K_hat + R`` with ``a = CKA(G, K)``: the part the reference explains (``seen``)
        and the residual, each projected on the PSD cone, plus the discarded spectral mass."""
        g, k = _centered_unit(self.G), _centered_unit(_as_array(reference))
        a = float((g * k).sum())
        resid = g - a * k
        vals, vecs = np.linalg.eigh(0.5 * (resid + resid.T))
        resid_psd = (vecs * np.maximum(vals, 0)) @ vecs.T
        return {"alignment": a,
                "seen": GramSpace(a * k, self.labels, self.predictions, name=f"{self.name} (seen)"),
                "residual": GramSpace(resid_psd, self.labels, self.predictions, name=f"{self.name} (residual)"),
                "discarded": discarded_mass(resid)}

    # ------------------------------------------------------------------ helpers
    def _labels(self, labels):
        labels = self.labels if labels is None else np.asarray(labels)
        if labels is None:
            raise ValueError("labels are required (pass them here or when building the space)")
        return labels


def _as_array(obj) -> np.ndarray:
    return obj.G if isinstance(obj, GramSpace) else np.asarray(obj, dtype=np.float64)


def _centered_unit(G: np.ndarray) -> np.ndarray:
    r = G.mean(1, keepdims=True)
    c = G - r - r.T + G.mean()
    f = np.linalg.norm(c)
    return c / f if f > 0 else c
