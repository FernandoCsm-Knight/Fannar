# Trees and other models

[`Interpreter`](../reference/interpreter.md) picks the sources from the type of the model:

| model | sources | "layers" |
|---|---|---|
| `torch.nn.Module` | `TorchSources`: activations, gradients, parameters | the modules in `layers` |
| scikit-learn tree (has `tree_` and `decision_path`) | `TreeSources`: path, slack, classes | each depth, `depth1 … depthD` |
| anything else | your `extractor` function | whatever your function returns |

## PyTorch models

Any classifier that returns logits `(N, K)` works: MLPs, CNNs, residual networks, transformers.
Name the modules to probe as they appear in `model.named_modules()`; with `layers=None`, every
direct child that has parameters is probed.

```python
interp = fa.Interpreter(model, X_test, y_test, layers=["layer1", "layer2", "layer3"],
                        reference_model=initial)
```

`X` can be an array, a tensor or a `DataLoader` (the first element of each batch is used).
The keyword arguments that `Interpreter` does not know are passed to `TorchSources`:

| option | default | meaning |
|---|---|---|
| `target` | `"fixed"` | class of the contrastive margin: `"fixed"`, `"predicted"` or an `int` |
| `fixed_class` | `0` | the class used by `target="fixed"` |
| `grid` | `4` | positions kept per spatial or token axis (`None` keeps all) |
| `layout` | `"channels"` | `"tokens"` for outputs shaped `(N, L, C)` |
| `sources` | all three | a subset of `("parameters", "activations", "gradients")` |
| `weight_modules` | automatic | `{layer: module or name}` whose weight defines the parameter source |
| `margin` | `"contrastive"` | or a callable `fn(logits, target) -> (N,)` |
| `device`, `batch_size` | model's device, `256` | |

### CNNs

A convolutional output `(N, C, H, W)` is average-pooled to `grid × grid` positions, so each
object has `C × grid²` values per source. For a map of where the model's energy sits inside one
image, use the positions of that image as the objects:

```python
emap = interp.energy_map(image, "layer2")          # (h, w) retained energy per position
interp.plot_energy_map(image, "layer2", image=rgb) # over the picture, values in [0, 1]
```

### Transformers

Modules that return a tuple are probed on their first tensor. For hidden states shaped
`(N, L, C)`, pass `layout="tokens"` so that the channels are the last axis:

```python
interp = fa.Interpreter(model, input_ids, y, layers=[f"encoder.layer.{t}" for t in range(6)],
                        layout="tokens", grid=8)
```

If your model takes several inputs (attention masks, token types) or returns an object instead of
a logits tensor, as Hugging Face models do, wrap it in a module whose `forward` takes one tensor
and returns the logits. The layer names are then prefixed by the wrapper's attribute name.

### The parameter source

The parameter source needs the weight `W` that produces the probed output. By default it is the
probed module's own weight or, inside a block, the last module up to it whose weight has as many
rows as the output has channels (`conv2` in a residual block, the `Linear` before a probed
activation). When none fits, the parameter source is skipped with a warning; point to the right
module with `weight_modules={"layer": "path.to.module"}`.

## Decision trees

A fitted scikit-learn `DecisionTreeClassifier` gets its own instantiation of the sources, one
"layer" per depth. At depth $l$ each object has:

- `path`: indicator of the nodes visited up to depth $l$ (the state of the tree);
- `slack`: $(x_f - t)/\sigma_f$ at every internal node already crossed, i.e. how far the object
  is from switching branch (the tree's sensitivity);
- `classes`: the class distribution of the node the object reaches at depth $l$.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
interp = fa.Interpreter(tree, X_test, y_test, feature_names=names,
                        scale=X_train.std(0))     # slack in standard deviations
report = interp.evaluate()
```

With trees, the default pipeline becomes `KernelPipeline(["path", "slack", "classes"])` and the
baseline is the path alone. `scale` and `depths` are passed to `TreeSources`.

## Any other model

Give `extractor`, a function that returns the sources of a batch of objects and the model's
predictions:

```python
def sources(X):
    layers = {"output": {"proba": model.predict_proba(X), "leaves": model.apply(X)}}
    return layers, model.predict(X)                 # optionally a third item: probabilities

interp = fa.Interpreter(model, X_test, y_test, extractor=sources,
                        pipeline=fa.KernelPipeline(["proba", "leaves"]),
                        baselines={})
```

The function can return `(layers, predictions)`, `(layers, predictions, probabilities)` or a
[`Sources`](../reference/sources.md) object. `layers` maps each layer name to
`{source name: array}`, with the objects on the first axis. The probabilities feed the soft
confusion of the `confusion_agreement` reading; without them the confusion is counted.

The default baseline is the `activations` source alone. It is skipped in the layers where your
function provides no `activations`, so pass `baselines={}` to say so explicitly, or baselines
built from the sources you do provide.

!!! note "Pair explanations with your own extractor"
    `explain_pair`, `explain_errors` and `swap_test` call your function on new rows (the object
    with one feature swapped), so it must work on any `(m, F)` float32 array, not only on
    `X_test`. `swap_test` also needs `predict=`, a function returning labels, for models that
    are neither PyTorch nor scikit-learn trees.
