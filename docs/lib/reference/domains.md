# Domain modules

One entry point per kind of data, each a subclass of [`Interpreter`](interpreter.md) that only
sets defaults and handles the inputs. The modules load on first access (`fa.tabular`, `fa.image`,
`fa.text`, `fa.tree`), so `import fannar` stays light.

## fannar.tabular

::: fannar.tabular.TabularInterpreter
    options:
      members: false

::: fannar.tabular.as_table

::: fannar.tabular.encode_labels

## fannar.image

::: fannar.image.ImageInterpreter
    options:
      members: false

## fannar.text

::: fannar.text.TextInterpreter
    options:
      members: false

::: fannar.text.wrap_hf

::: fannar.text.HFClassifier
    options:
      members: false

::: fannar.text.block_names

## fannar.tree

::: fannar.tree.TreeInterpreter
    options:
      members: false
