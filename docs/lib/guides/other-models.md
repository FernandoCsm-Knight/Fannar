# Choosing a module

Since 0.4.0 there is one entry point per kind of data. Each is a thin
[`Interpreter`](../reference/interpreter.md) with sensible defaults and the same methods, and it
gives the same numbers as `fa.Interpreter` with the same options:

| data / model | module | class | guide |
|---|---|---|---|
| PyTorch model on tabular rows | `fannar.tabular` | `TabularInterpreter` | [Tabular data](tabular.md) |
| PyTorch CNN or ViT on images | `fannar.image` | `ImageInterpreter` | [Images](image.md) |
| PyTorch transformer on tokens (Hugging Face included) | `fannar.text` | `TextInterpreter` | [Text](text.md) |
| scikit-learn decision tree | `fannar.tree` | `TreeInterpreter` | [Decision trees](tree.md) |
| anything else | `fannar` | `Interpreter(extractor=...)` | [below](#any-other-model) |

```python
import fannar as fa
fa.tabular.TabularInterpreter    # the modules load on first access; no extra import needed
```

`fa.Interpreter` still picks the sources from the type of the model, as in earlier versions:

| model | sources | "layers" |
|---|---|---|
| `torch.nn.Module` | `TorchSources`: activations, gradients, parameters | the modules in `layers` |
| scikit-learn tree (has `tree_` and `decision_path`) | `TreeSources`: path, slack, classes | each depth, `depth1 … depthD` |
| anything else | your `extractor` function | whatever your function returns |

## PyTorch options

These options apply to `TabularInterpreter`, `ImageInterpreter`, `TextInterpreter` and to
`fa.Interpreter` with a PyTorch model. Any classifier that returns logits `(N, K)` works.
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
| `grid` | `4` (`8` in `TextInterpreter`) | positions kept per spatial or token axis (`None` keeps all) |
| `layout` | `"channels"` (`"tokens"` in `TextInterpreter`) | `"tokens"` for outputs shaped `(N, L, C)` |
| `sources` | all three | a subset of `("parameters", "activations", "gradients")` |
| `weight_modules` | automatic | `{layer: module or name}` whose weight defines the parameter source |
| `margin` | `"contrastive"` | or a callable `fn(logits, target) -> (N,)` |
| `device`, `batch_size` | model's device, `256` | |

### The parameter source

The parameter source needs the weight `W` that produces the probed output. By default it is the
probed module's own weight or, inside a block, the last module up to it whose weight has as many
rows as the output has channels (`conv2` in a residual block, the `Linear` before a probed
activation). When none fits, the parameter source is skipped with a warning; point to the right
module with `weight_modules={"layer": "path.to.module"}`.

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
