# The representation space

A [`GramSpace`](../reference/space.md) holds the final matrix $G$ of $n$ objects and everything
read from it. You get one from a pipeline, from `interp.space(layer)`, or directly from any
symmetric PSD matrix:

```python
space = fa.GramSpace(G, labels=y, predictions=pred)
space.n, space.is_psd(), space.G
```

The input is symmetrised. Labels and predictions are optional and only stored for the readings
and the plots.

## Geometry

$G$ defines the pseudometric $d(i, j)^2 = G_{ii} + G_{jj} - 2G_{ij}$: the distance between the
implicit feature vectors $\phi_i$ and $\phi_j$.

```python
space.distances               # (n, n) matrix, computed once
space.distance(i, j)
space.neighbors(i, k)         # the k nearest objects to i, nearest first
space.level_band(i, r, delta) # Γ_{r,δ}(i) = { j : |d(i, j) − r| ≤ δ }
space.level_set(i, r)         # Γ_r(i), up to a tolerance
space.equivalence_classes()   # groups of objects the space cannot tell apart (d = 0)
space.subset(idx)             # the sub-space of some objects (not re-transformed)
```

It is a *pseudo*metric: two different objects can be at distance zero. That happens, for
instance, in a tree layer where two objects took the same path with the same slack.
`equivalence_classes()` lists those groups.

## Spectrum and energy

```python
space.eigenvalues             # descending, clipped at zero
space.eigenvectors            # columns in the same order
space.eigenvalues_raw         # ascending, unclipped: negative values reveal a non-PSD input
space.participation_rank()    # round((tr G)² / ‖G‖²_F), O(n²)
space.rank("tau", tau=0.95)   # modes holding 95% of the energy
space.mode_energy()           # (n, n): E_ik = λ_k V_ik², rows sum to G_ii
```

`energies()` splits each object's energy between the retained subspace $V_r$ and its orthogonal
complement:

```python
e = space.energies()          # r = participation rank; or energies(5), energies("tau", 0.9)
e["rank"]                     # r
e["E_par"], e["E_perp"]       # retained and orthogonal energy per object
e["E_total"]                  # = diag(G)
e["E_ratio"]                  # E_perp / E_total
e["retained_fraction"]        # share of the trace in the first r modes
```

The `rank` argument accepts `"participation"` (default), `"tau"` or an explicit integer.

## Embedding

```python
coords, retained = space.embedding(2)   # sqrt(λ_k) V_ik on the two dominant modes
```

This is kernel PCA of $G$: with all modes, `coords @ coords.T` reproduces $G$. `retained` is the
fraction of the trace the shown modes carry; read a 2-D picture with that number in mind.

## Classes

```python
space.containment(labels)     # in [0, 1]
```

The fraction of the centred class-indicator directions that lies inside $V_r$. By default
$r = K - 1$ (number of classes minus one), which makes the value comparable across spaces; the
participation rank of each space is not.

## Comparing spaces

```python
space.cka(other)              # centred kernel alignment, other: GramSpace or array
parts = space.decompose(K)    # Ĝ = a·K̂ + R
parts["alignment"]            # a = CKA(G, K)
parts["seen"]                 # GramSpace of the part K explains
parts["residual"]             # GramSpace of what K does not explain (projected on the PSD cone)
parts["discarded"]            # spectral mass lost in that projection
```

`decompose` answers "what does this space contain beyond the reference?". With `K` the
activations-only Gram, for example, the residual is the part of the joint space that the
activations alone do not explain.

## Plots

```python
from fannar import plots

plots.plot_gram(space, order_by=y)   # components, their product, and G
plots.plot_spectrum(space)           # eigenvalues, cumulative energy, participation rank
```
