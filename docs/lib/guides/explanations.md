# Explanations

Beyond the readings on whole spaces, Fannar explains individual decisions **by contrast**: why
does the model place object $i$ apart from object $j$? Pair explanations need vector inputs,
`X` of shape `(n, F)`.

## Pairs

For a pair $(i, j)$ and each feature $f$, let $x_i^{(f)}$ be $x_i$ with feature $f$ replaced by its
value in $x_j$. The score of $f$ in a layer is

$$
\Delta d_f = d(i, j) - d\big(i^{(f)}, j\big),
$$

measured in that layer's representation space. A positive $\Delta d_f$ means the swap brings $i$
closer to $j$: the feature is part of what separates them for the model. Only the $F + 2$ rows
$x_j, x_i, x_i^{(1)}, \dots, x_i^{(F)}$ go through the model, so the cost is linear in the number
of features.

```python
expl = interp.explain_pair(i)          # j = nearest object (in input space) with another prediction
expl = interp.explain_pair(i, j)       # or an explicit partner
expl.j, expl.pred_i, expl.pred_j

expl.top(5)                            # [(feature, x_i, x_j, share), ...], last layer
expl.top(5, layer="layer2")
expl.ranking(), expl.shares()          # indices, and positive scores normalised to sum 1
expl.scores["layer2"]                  # raw Δd per feature
expl.to_frame()                        # pandas
expl.plot(k=8)                         # share per layer (heatmap) and the values of i and j
```

!!! warning "First order"
    $\Delta d_f$ swaps one feature at a time. Features that only matter together with others get
    no credit.

## Errors

`explain_errors` aggregates pair explanations over the model's mistakes. Each misclassified
object $i$ is paired with the nearest object the model gets right **in the true class of $i$**,
so the shares say which features keep the error away from where it belongs.

```python
errs = interp.explain_errors()                 # against="correct" (default) or "unlike"
errs = interp.explain_errors(max_errors=200, reference=100, seed=0)

errs.top(5)                    # [(feature, mean share, fraction of errors where it is first)]
errs.mean_shares()             # per feature, over the errors
errs.reference_shares()        # the same over correctly predicted objects: the general importance
errs.direction()               # +1: errors have lower values than their partners; −1: higher
errs.confusions()              # [(true, predicted, count)], most frequent first
errs.by_class(), errs.by_confusion()
errs.pairs                     # [(error, partner)], e.g. for swap_test
errs.to_frame()
errs.plot(rows="class")        # or rows="confusion"
```

Compare `mean_shares()` with `reference_shares()`: a feature with a high share among the errors
but a low one in general is specific to the mistakes.

## Validating with the swap test

The pair ranking is validated outside the geometry. Copy the top-$k$ features of $j$ into $i$,
for $k = 0, \dots, F$, and check whether the model's prediction becomes the prediction of $j$; the
area under that curve is higher when the ranking finds the decisive features sooner. This is the
procedure of nearest-instance counterfactuals such as NICE.

```python
interp.swap_test(50, layer="layer3")
# {"method": <area>, "absdiff": <area>, "random": <area>, "pairs": 50, "layer": "layer3"}

interp.swap_test(errs.pairs[:30])     # explicit pairs
```

`method` ranks by $\Delta d$, `absdiff` by $|x_j - x_i|$, and `random` shuffles the features that
differ. The method's ranking is useful where it beats both. With an integer, the pairs are random
correctly predicted objects, each with its nearest object of another prediction.

## Energy maps

For a single input with spatial structure (an image, a series, a sequence of tokens), the
**positions** of that input can be the objects. `energy_map` builds the space of the positions of
one layer and returns the retained energy $E_\parallel$ of each one:

```python
emap = interp.energy_map(x, "layer2", max_side=32)   # (h, w) for images, (L,) for sequences
interp.plot_energy_map(x, "layer2", image=rgb)       # rgb: (H, W[, 3]) in [0, 1]
```

The components use linear kernels without the angular transform, which would give every
position the same energy (`fa.positions_pipeline()`). `max_side` caps the grid, since the Gram
is $P \times P$ for $P$ positions. Energy maps need a PyTorch model.
