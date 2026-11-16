# Readings and reports

A reading relates a representation space to the **behaviour of the model**: its predictions,
its errors, its confusions. The claim under test is that the geometry reflects what *this model*
does, not that it separates the dataset's classes, so the reference is always something the
model produced.

## The readings

| reading | question | range |
|---|---|---|
| `knn_prediction` | Do an object's $k$ nearest neighbours share its prediction? (leave-one-out, against the model's predictions) | $[0, 1]$ |
| `correctness_auc` | Does the geometric margin anticipate where the model is wrong? | AUC, 0.5 = chance |
| `confusion_agreement` | Are the classes the model confuses close to each other? (Spearman, $\ge 5$ classes) | $[-1, 1]$ |
| `containment` | Are the class directions inside the retained subspace $V_r$, $r = K - 1$? | $[0, 1]$ |
| `e_ratio_error_auc` | Does the energy outside $V_r$ signal the errors? | AUC, 0.5 = chance |
| `rank` | participation rank of the space (context, not a reading) | |

**Geometric margin.** For object $i$, with $\bar d_{\text{own}}$ the mean distance to the
objects of its *predicted* class and $\bar d_{\text{near}}$ the mean distance to the closest
other class,

$$
\text{margin}(i) = \frac{\bar d_{\text{near}} - \bar d_{\text{own}}}{\bar d_{\text{near}} + \bar d_{\text{own}}} \in [-1, 1].
$$

The margin groups objects by the model's predictions; the true labels only define the target
(right or wrong). Using the labels in the margin too would make the reading circular, and would
rely on a signal that does not exist when the model is deployed.

**Confusion.** With probabilities, the confusion between classes $a$ and $b$ is the mean
probability the model gives to $b$ on objects of $a$, symmetrised. It is informative even when the
model rarely errs. Without probabilities, the errors are counted.

Readings that need errors (`correctness_auc`, `e_ratio_error_auc`) are `NaN` when the model makes
fewer than 5 errors, or fewer than 5 correct predictions, on the sample.

## Reports

`interp.evaluate()` computes every reading for every layer and representation and returns a
[`Report`](../reference/report.md):

```python
report = interp.evaluate(layers=None, include_baselines=True, include_floor=True)

report.layers                 # ["layer1", "layer2", ...]
report.representations        # ["joint", "activations", "floor"]
report.values["joint"]["layer2"]["correctness_auc"]
report.series("correctness_auc", "joint")         # one value per layer
report.above_floor("correctness_auc")             # bool per layer (None without a floor)
report.meta                   # accuracy, n, pipeline
```

| representation | what it is |
|---|---|
| `joint` | the interpreter's pipeline (by default, the paper's three sources) |
| one per baseline | each pipeline in `baselines` (by default, `activations` alone) |
| `floor` | the joint pipeline on `reference_model`, against the trained model's behaviour |

!!! floor "Read against the floor"
    A reading is evidence about training only where it is above its floor. Untrained networks
    already produce structured spaces, and some readings are far from chance on them.

### Exporting

```python
print(report.to_markdown(digits=3))   # one table per reading, layers as columns
report.to_json("report.json")         # also returns the text
df = report.to_frame()                # long format: representation, layer, reading, value
report.plot()                         # same as interp.plot_report(report)
```

### Baselines

Any pipeline or preset name can be a baseline. Use a dictionary to name them:

```python
interp = fa.Interpreter(model, X, y, layers=layers,
                        baselines={"acts": "activations", "grads": "gradients",
                                   "euclid": "euclidean"})
```

## Readings on any space

The readings also work on a space you built yourself:

```python
from fannar import readings

values = fa.evaluate(space, labels, predictions, probabilities=None, k=5)
readings.correctness_auc(space, labels, predictions)
readings.geometric_margin(space, predictions)        # per object
readings.class_distance_matrix(space, labels)        # (K, K) mean distances, classes
```
