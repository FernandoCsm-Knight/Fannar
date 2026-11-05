# The method

Fannar describes what a trained model does with a set of **objects** (test samples, positions
of an image, tokens) through a single positive semidefinite matrix $G \in \mathbb{R}^{n \times n}$,
built from several **sources** of information about each object.

## The construction

$$
G \;=\; T\big(\,T_1(K_1) \circ T_2(K_2) \circ \cdots \circ T_d(K_d)\,\big)
$$

1. **Sources.** For every object $i$ and source $l$ there is an array $x_i^{(l)}$: the
   activations of a layer, a gradient, the path through a tree, or anything else.
2. **Kernels.** Each source gets a positive-definite kernel $\kappa_l$, which gives the raw Gram
   matrix $K_l[i, j] = \kappa_l(x_i^{(l)}, x_j^{(l)})$. You may also supply $K_l$ directly.
3. **Component transforms.** Each $K_l$ goes through an *admissible* transformation $T_l$.
4. **Hadamard composition.** The components are multiplied entry by entry.
5. **Final transform.** An admissible $T$ normalises the product.

A transformation is **admissible** when it maps symmetric PSD matrices to symmetric PSD matrices
and does not depend on the order of the objects, $T(PGP^\top) = P\,T(G)\,P^\top$ for every
permutation $P$. Admissible transformations are closed under composition, and by the Schur
product theorem the Hadamard product of PSD matrices is PSD. So $G$ is PSD for *any* choice of
kernels and admissible transforms: there are vectors $\phi_i$ with $G_{ij} = \langle \phi_i,
\phi_j \rangle$, and everything below is computed from $G$ alone, never from the $\phi_i$.

### The instantiation of the paper

The default pipeline (`"paper"`, or [`fa.paper_pipeline()`](reference/pipeline.md)) uses, for
each source:

- the linear kernel $\kappa(x, y) = \langle x, y \rangle$;
- the angular transform $T_a(K) = D^{-1/2} K D^{-1/2}$ (cosine similarity, unit diagonal),
  followed by a shift $\tfrac12 (T_a(K) + J)$, which moves the cosine into $[0, 1]$ and keeps
  the product of several components from collapsing to the identity;

and, after the Hadamard product, centring followed by trace normalisation, $T = T_{tr} \circ T_c$.

!!! tip "Why linear kernels"
    With linear-type kernels, the Hadamard product realises the tensor product of the feature
    spaces, so the joint similarity keeps the *multiplicative interaction* between sources.
    Gaussian kernels lose it: the product of Gaussians is the Gaussian of the summed exponents,
    and the composition becomes an additive combination of the source metrics. `RBF` is
    available, but `Linear` and `Cosine` are the defaults for this reason.

## Sources of a neural network

For a PyTorch classifier, [`TorchSources`](reference/sources.md) extracts three sources at every
probed layer, with one forward and one backward pass per batch:

| source | what it is |
|---|---|
| `activations` | the layer output $A$, pooled to a common grid of positions |
| `gradients` | the derivative of the contrastive margin $m = f_c - \frac{1}{K-1}\sum_{c' \neq c} f_{c'}$ with respect to that output |
| `parameters` | the filters the object activated, $p[:, s] = \sum_c A[c, s]\, W[c, :]$, with $W$ the weight that produces the output |

Outputs of shape `(N, C)` give one position; `(N, C, H, W)` is pooled to `grid × grid`
positions; `(N, L, C)` (tokens, with `layout="tokens"`) is pooled over `L`. The default is
`grid=4`.

Activations say *what* the layer represents, gradients *how the decision depends* on it, and
parameters *which part of the learned weights* the object uses. Each source alone gives a
different space; the product keeps only the similarity the three agree on.

## Two cautions the library enforces

!!! warning "The target of the gradient"
    With `target="predicted"`, the gradient of each object is taken towards its own prediction,
    so the space already carries the decision and any reading about the decision becomes
    circular. The default is `target="fixed"`: the margin of the same class (`fixed_class`, 0 by
    default) for every object.

!!! floor "The floor"
    Every reading needs a reference. Pass `reference_model`, the same architecture with its
    initial weights: the report then measures the geometry of that untrained network against the
    behaviour of the trained one. The only difference between a reading and its floor is
    training, so a reading below or at its floor says nothing about what was learned.

Untrained networks already have structure, so geometric readings on them can be far from
chance. Without a floor it is impossible to tell structure that training created from
structure the architecture and the data already had.

## What is read from G

**Geometry.** $G$ defines a pseudometric

$$
d(i, j)^2 = G_{ii} + G_{jj} - 2\,G_{ij},
$$

and with it neighbourhoods, level sets $\Gamma_r(i) = \{ j : d(i, j) = r \}$ and classes of
objects the space cannot tell apart ($d = 0$).

**Spectrum.** With $G = V \Lambda V^\top$, the energy of object $i$ in mode $k$ is
$E_{ik} = \lambda_k V_{ik}^2$, and each row sums to $G_{ii}$. The first $r$ modes form the
**retained subspace** $V_r$; by default $r$ is the participation rank

$$
r = \operatorname{round}\!\left( \frac{(\operatorname{tr} G)^2}{\lVert G \rVert_F^2} \right),
$$

which needs no eigendecomposition. Each object then splits into retained energy $E_\parallel$
and orthogonal energy $E_\perp$, and

$$
E_{\text{ratio}}(i) = \frac{E_\perp(i)}{G_{ii}}
$$

measures how much of the object lies outside the dominant structure of the space. The reading
`e_ratio_error_auc` tests whether a high $E_{\text{ratio}}$ signals where the model errs.

**Readings.** The readings relate this geometry to what the model *does* (its predictions,
errors and confusions), never only to the dataset's labels. See
[Readings and reports](guides/readings.md).
