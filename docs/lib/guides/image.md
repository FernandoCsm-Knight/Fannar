# Images

`fannar.image` interprets PyTorch image classifiers (CNNs, residual networks, vision transformers
whose blocks return `(N, C, H, W)`). `ImageInterpreter` is an
[`Interpreter`](../reference/interpreter.md) with defaults for images.

```python
from fannar.image import ImageInterpreter

interp = ImageInterpreter(
    model, test_loader, y_test,               # array, tensor or DataLoader
    layers=["layer1", "layer2", "layer3"],
    reference_model=initial,
    class_names=classes,
)
report = interp.evaluate()
interp.plot_embedding()
```

`X` can be an array, a tensor or a `DataLoader` (the first element of each batch is used, so
`(image, label)` batches work as they are).

## Pooling

A convolutional output `(N, C, H, W)` is average-pooled to `grid × grid` positions, so each object
has `C × grid²` values per source. `ImageInterpreter` uses `grid=4`. A larger grid keeps more
spatial detail, but the size of the sources (and the [cost](cost.md)) grows with `grid²`.
`grid=None` keeps every position.

## Where the energy sits inside one image

With images as objects there are no per-feature explanations (`explain_pair` needs vector
inputs). To see which parts of one image carry the model's energy, use the positions of that
image as the objects:

```python
emap = interp.energy_map(image, "layer2")           # (h, w) retained energy per position
interp.plot_energy_map(image, "layer2", image=rgb)  # over the picture, values in [0, 1]
```

`max_side` (default 32) caps the grid of positions, since the Gram matrix is `P × P`. The map has
the resolution of the probed layer: after two stride-2 stages a block's map is 4 times smaller per
side than the input, and `plot_energy_map` stretches it over the picture.

## Full example

[`examples/image_digits_cnn.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/image_digits_cnn.py)
trains a three-block CNN on the 8×8 `digits` and draws the energy map of one image. See
[Examples](../examples.md#images-and-where-the-energy-sits).

## Things to watch

- **Name the blocks.** The layers are module names. With `nn.Sequential` and anonymous children
  they are `"0"`, `"1"`… Name the blocks (`add_module("conv1", ...)`) or look them up in
  `dict(model.named_modules())`.
- **Probe the block, not a lone ReLU.** Probing a block's output gives the state the next block
  reads. The parameter source is found inside the block (the last convolution with as many
  output channels). If it is not, pass `weight_modules={"layer": "path.to.conv"}`.
- **Batch norm and dropout.** The extraction puts the model in `eval()` mode and restores it
  afterwards, so batch statistics are not updated.
- **Memory.** The sources hold `n × C × grid²` values per layer and source. For wide layers,
  lower `grid` or the number of objects before raising anything else.
