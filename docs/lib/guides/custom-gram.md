# Your own Gram matrices

The construction does not depend on neural networks. Any set of objects with one or more sources
of information can go through a [`KernelPipeline`](../reference/pipeline.md), and any component
can be your kernel, your transform, or a matrix you already computed. The full script is
[`examples/custom_gram.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/custom_gram.py).

## A pipeline

```python
import numpy as np
import fannar as fa

pipe = fa.KernelPipeline(
    {
        "state":  fa.Component(fa.Linear(), fa.angular() >> fa.shift(1.0)),
        "text":   fa.Component(fa.RBF(gamma=0.1), fa.identity),
        "mine":   fa.Component(lambda A, B: (A @ B.T + 1) ** 2),   # any function k(X, Y)
    },
    final=fa.center >> fa.trace_normalize,
)

space = pipe(sources={"state": S, "text": T, "mine": M}, labels=y)
```

Each source is an array whose first axis indexes the objects; the other axes are flattened.
NumPy arrays, PyTorch tensors and lists are accepted.

There are three shorter ways to declare the components:

```python
fa.KernelPipeline(["a", "b"])                     # default component for each name
fa.KernelPipeline({"a": "linear", "b": "cosine"}) # a kernel name per source
fa.KernelPipeline({"a": fa.Polynomial(degree=3)}) # a kernel object per source
```

The default component is the one of the paper: linear kernel, then `angular() >> shift(1.0)`.
The default final transform is `center >> trace_normalize`.

## Matrices you already have

A raw Gram passed in `grams` takes precedence over the kernel applied to `sources`, so you can
replace any component, or all of them:

```python
space = pipe(sources={"state": S, "text": T}, grams={"mine": K_mine})   # mix
space = pipe(grams={"state": K1, "text": K2, "mine": K3})               # matrices only
```

The component transform is still applied to a matrix given in `grams`. To skip the construction
altogether, build the space directly from any PSD matrix:

```python
space = fa.GramSpace(my_psd_matrix, labels=y)
```

With an [`Interpreter`](../reference/interpreter.md), `interp.space_from_grams({...})` does the
same with the model's labels and predictions attached.

## Kernels

| kernel | $\kappa(x, y)$ | notes |
|---|---|---|
| `Linear()` | $\langle x, y \rangle$ | with `angular()`, the cosine |
| `Cosine(eps=1e-12)` | $\langle x, y \rangle / (\lVert x \rVert \lVert y \rVert)$ | zero vectors get similarity 0 (1 with themselves) |
| `RBF(gamma=None)` | $\exp(-\gamma \lVert x - y \rVert^2)$ | `gamma=None` uses the median heuristic |
| `Polynomial(degree=2, coef0=1, gamma=None)` | $(\gamma \langle x, y \rangle + c_0)^{p}$ | `coef0 >= 0`; `gamma` defaults to `1/D` |
| `FunctionKernel(fn, name=None)` | `fn(X, Y)` | any callable; positive definiteness is up to you |

A plain function `fn(X, Y) -> (n, m) array` is wrapped in `FunctionKernel` automatically. For a
reusable kernel, subclass `Kernel` and implement `compute` on `(n, D)` and `(m, D)` arrays:

```python
class Laplacian(fa.Kernel):
    name = "laplacian"

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def compute(self, X, Y):
        d = np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
        return np.exp(-self.gamma * d)
```

!!! tip
    Prefer `Linear` or `Cosine` when you want the interaction between sources: under the Hadamard
    product, Gaussian components add their metrics instead of multiplying the similarities. See
    [The method](../concepts.md#the-instantiation-of-the-paper).

## Transforms

Transforms compose with `>>`, from left to right: `fa.center >> fa.trace_normalize` centres and
then normalises.

| transform | effect | cost |
|---|---|---|
| `identity` | $G$ | |
| `center` | $HGH$, with $H = I - \frac1n J$ | $O(n^2)$ |
| `trace_normalize` | $G / \operatorname{tr} G$ | $O(n)$ |
| `frobenius_normalize` | $G / \lVert G \rVert_F$ | $O(n^2)$ |
| `max_eig_normalize` | $G / \lambda_{\max}$ | $O(n^3)$ |
| `angular(eps=1e-12)` | $D^{-1/2} G D^{-1/2}$ (unit diagonal; 0 for null objects) | $O(n^2)$ |
| `shift(c=1.0)` | $(G + cJ)/(1 + c)$, `c >= 0` | $O(n^2)$ |
| `whiten(tol=1e-10)` | projector onto the range of $G$ | $O(n^3)$ |
| `spectral(f)` | $V f(\Lambda) V^\top$, with $f \ge 0$ on $\lambda \ge 0$ | $O(n^3)$ |
| `psd_project()` | nearest PSD matrix (negative eigenvalues set to 0) | $O(n^3)$ |

`identity`, `center`, `trace_normalize`, `frobenius_normalize` and `max_eig_normalize` are
transform objects; the others are functions that return one, so they take parentheses.

A transform of your own is any function of a symmetric matrix wrapped in `Transform`. It is your
responsibility that it is admissible (PSD in, PSD out, independent of the order of the objects):

```python
sqrt_spectrum = fa.spectral(np.sqrt, name="sqrt")         # V Λ^{1/2} Vᵀ
square = fa.Transform(lambda G: G * G, "hadamard_square")  # PSD by the Schur product theorem

component = fa.Component(fa.Linear(), fa.angular() >> square)
```

Entry-wise functions are the usual trap: most of them (`np.log1p`, `np.sqrt` entry by entry) do
not keep a matrix PSD.

## Presets

| preset | what it is |
|---|---|
| `fa.paper_pipeline(sources=("parameters", "activations", "gradients"), shift_by=1.0)` | the instantiation of the paper |
| `fa.single_source("activations", angular_shift=True)` | one source with the paper's component transform, for ablations |
| `fa.single_source(..., angular_shift=False)` | the plain linear centred Gram: Euclidean distances, the geometry of linear CKA and RSA |
| `fa.positions_pipeline(sources)` | positions of one input as objects (used by the energy maps) |

Wherever a pipeline is expected, the names in `fa.PRESETS` also work: `"paper"`,
`"activations"`, `"gradients"`, `"parameters"`, `"euclidean"`.

## The Hadamard root

`KernelPipeline(..., root=True)` takes the $d$-th Hadamard root of the product of $d$ components
before the final transform: the entry-wise geometric mean of the components. Fractional Hadamard powers do not preserve positive semidefiniteness, so the result is
projected back on the PSD cone, and the fraction of spectral mass that projection discarded is
stored in `space.info["root_discarded"]`.

## Inspecting a pipeline

```python
print(pipe)                       # kernels and transforms of every component
grams = pipe.component_grams(sources={...})   # T_l(K_l) for each component
space.components                  # the same, kept on the space (unless keep_components=False)
space.info["pipeline"]
```
