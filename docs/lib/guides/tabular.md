# Tabular data

`fannar.tabular` interprets a PyTorch model that reads rows of features (an MLP, a residual MLP,
a model for tabular data). `TabularInterpreter` is an [`Interpreter`](../reference/interpreter.md)
that accepts pandas objects directly. The construction of the space is the same.

```python
from fannar.tabular import TabularInterpreter

interp = TabularInterpreter(
    model, df_test, y_test,                   # DataFrame and labels
    layers=["block1", "block2", "block3"],    # names from model.named_modules()
    reference_model=initial,                  # the same network before training (the floor)
)
report = interp.evaluate()
interp.explain_errors().top(5)                # features named after df_test.columns
interp.explain_pair(0).top(5)
interp.swap_test(30)
```

What changes compared with `fa.Interpreter`:

| input | what `TabularInterpreter` does |
|---|---|
| `X` is a `DataFrame` | its columns become `feature_names`; the values are converted to `float32` |
| `y` is a categorical `Series` | the category codes are the labels and the categories are `class_names` |
| `y` holds strings | encoded by sorted unique value (the order of scikit-learn's `LabelEncoder`) |
| `y` is numeric | used as is: the model's output indices `0 … K-1` |

Explicit `feature_names=` and `class_names=` take precedence. The label encoding must match the
one used to train the model. Pass integer labels when in doubt.

Nothing is preprocessed. Pass the rows in the scale the model was trained on (for instance,
already standardised). Every other keyword (`target`, `fixed_class`, `device`, `batch_size`,
`pipeline`, `baselines`, `k`) works as in [`Interpreter`](../reference/interpreter.md) and
[PyTorch options](other-models.md#pytorch-options).

## Full example

[`examples/tabular_wine.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/tabular_wine.py)
trains an MLP on scikit-learn's `wine` and walks through pandas input, pair and error explanations
and the swap test. Its output is discussed in [Examples](../examples.md#a-table-with-pandas).

## Things to watch

- **Scale.** The pair explanations swap raw values between rows. If the model was trained on
  standardised features, pass standardised rows, and the values in `top()` are standardised too.
- **Label order.** With string labels, the sorted order must be the one used to train the model.
  When the model was trained with its own encoding, pass integer labels and
  `class_names=` explicitly.
- **Errors.** `explain_errors()` raises `ValueError` when the model makes no errors on `X`. Use a
  held-out set; the training rows of a well-fitted model may have none.
- **Cost.** Each pair runs $F + 2$ rows through the model, so explaining every error of a large
  test set with many features takes a while. Cap it with `max_errors=` (see [Cost](cost.md)).

For a scikit-learn decision tree on the same table, use [`fannar.tree`](tree.md).
