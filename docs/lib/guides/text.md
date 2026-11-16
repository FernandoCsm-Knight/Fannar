# Text and token sequences

`fannar.text` interprets PyTorch models whose hidden states are token sequences `(N, L, C)`:
transformer encoders, Hugging Face classifiers, or any model with a sequence axis.
`TextInterpreter` is an [`Interpreter`](../reference/interpreter.md) that reads the channels on
the last axis (`layout="tokens"`) and pools the sequence to `grid=8` positions.

## Hugging Face models

Hugging Face models take several inputs and return an object rather than a logits tensor.
`wrap_hf` turns one into a module whose `forward(input_ids)` returns the logits and builds the
attention mask from the padding token:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fannar.text import TextInterpreter, wrap_hf

tokenizer = AutoTokenizer.from_pretrained(name)
hf_model = AutoModelForSequenceClassification.from_pretrained(name).eval()
ids = tokenizer(texts, padding="max_length", truncation=True, max_length=128,
                return_tensors="pt")["input_ids"]

model = wrap_hf(hf_model, pad_token_id=tokenizer.pad_token_id)
interp = TextInterpreter(model, ids, labels)        # layers: every encoder block
report = interp.evaluate()
```

`transformers` is not a dependency of Fannar. `wrap_hf` only needs the call
`model(input_ids=..., attention_mask=...)` to return an object with `.logits`, a tuple or a tensor.
Extra keyword arguments of `wrap_hf` are passed to every call of the model.

## Which layers

With `layers=None`, the probed layers are the blocks of the model's longest `nn.ModuleList`
(`encoder.layer.*`, `transformer.h.*`, `layers.*`…), as returned by `fannar.text.block_names`:

```python
from fannar.text import block_names
block_names(model)    # ['model.bert.encoder.layer.0', ..., 'model.bert.encoder.layer.11']
```

The wrapper stores the original model as `model`, so the names gain the prefix `model.`. Pass
`layers=` to probe a subset. Blocks that return a tuple are probed on their first tensor.

## Pooling and token energy

Each hidden state is average-pooled along the sequence to `grid` positions. `grid=None` keeps every
token, which requires sequences of the same length. `interp.energy_map(ids[0], layer)` returns the
retained energy of every token of one sequence (the tokens are the objects). Slice off the padding
before reading it: `energy[: (ids[0] != pad).sum()]`.

!!! tip "Early and middle blocks for token maps"
    In the [text example](../examples.md#text-layer-by-layer), the keyword that decides the class
    is the most energetic token in 54% and 66% of the sentences in the first two blocks (chance
    is about 9%). In the last block it is 12%, because attention has already spread the
    information over every token. Token maps answer "which tokens carry the information" best
    before the last blocks.

Your own model only has to take one tensor and return logits `(N, K)`. In that case
`TextInterpreter(model, X, y, layers=[...])` works without `wrap_hf`.

## Full example

[`examples/text_keywords.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/text_keywords.py)
trains a three-block transformer on synthetic sentences, with the Hugging Face interface, and
compares the token energy across blocks. See [Examples](../examples.md#text-layer-by-layer).

## Things to watch

- **Padding.** Without `pad_token_id`, `wrap_hf` attends to every token, padding included, which
  is not how the model was trained. Pass the tokenizer's `pad_token_id`.
- **Same length.** The objects go through the model in batches, so the sequences must be padded
  to the same length (`padding="max_length"` in the tokenizer).
- **Large models.** Each layer and source keeps `n × C × grid` values. A 12-block model with
  `C = 768` and `grid=8` stores about 6 000 numbers per object, layer and source. Probe a subset
  of `layers=` and a few hundred objects first.
- **Reference model.** For the floor, wrap the untrained copy too:
  `reference_model=wrap_hf(initial, pad_token_id=...)`, so its layer names match.
