# Decision trees

`fannar.tree` interprets a fitted scikit-learn `DecisionTreeClassifier`. A tree has no activations
and no gradient, so it gets its own instantiation of the sources, one "layer" per depth. At depth
$l$ each object has:

- `path`: indicator of the nodes visited up to depth $l$ (the state of the tree);
- `slack`: $(x_f - t)/\sigma_f$ at every internal node already crossed, i.e. how far the object
  is from switching branch (the tree's sensitivity);
- `classes`: the class distribution of the node the object reaches at depth $l$.

```python
from sklearn.tree import DecisionTreeClassifier
from fannar.tree import TreeInterpreter

tree = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
interp = TreeInterpreter(tree, df_test, y_test, X_train=df_train)   # slack in training std units
report = interp.evaluate()
interp.explain_pair(0).top(5)
interp.explain_errors().top(5)
```

The joint pipeline is `KernelPipeline(["path", "slack", "classes"])` and the baseline is the path
alone.

| argument | meaning |
|---|---|
| `X_train` | training rows. Their per-feature standard deviation is $\sigma_f$ in the slack |
| `scale` | an explicit per-feature $\sigma_f$ (overrides `X_train`). Without either, $\sigma_f = 1$ |
| `depths` | the depths used as layers (default `1 … max_depth`) |
| `reference_model` | another tree (e.g. shallower or randomised) used as the floor |

`X` can be a `DataFrame` (its columns become `feature_names`). The labels can be any of the tree's
classes, strings included. They are mapped to indices into `tree.classes_`, which also become
`class_names`, so the readings and explanations work for any label type.

`fa.Interpreter(tree, X, y, scale=...)` still works as in earlier versions.

## Full example

[`examples/tree_wine.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/tree_wine.py)
fits a depth-4 tree on `wine` with text labels and a tree fitted on shuffled labels as the floor.
See [Examples](../examples.md#the-same-table-with-a-tree).

## Things to watch

- **`X_train` for the slack.** Without it, the slack is in the raw units of each feature. A
  feature measured in thousands then dominates the distance to the thresholds. Pass the training
  rows, or `scale=`.
- **A floor for trees.** There is no "untrained tree". A tree of the same depth fitted on shuffled
  labels is a useful floor: same structure, no signal.
- **Depth 1.** The first layer has a single split. The path alone puts every object at one of
  two points, and only the slack spreads them. Expect the joint and `path` readings to differ
  little there, and more from depth 2 on.
- **Ensembles.** Random forests and boosting are not single trees. Use `fa.Interpreter` with an
  `extractor` built from `model.apply(X)` (see [Choosing a module](other-models.md#any-other-model)).
