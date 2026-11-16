# Transforms

::: fannar.transforms
    options:
      show_root_heading: false
      members: false

::: fannar.Transform

## Constant transforms

| name | effect |
|---|---|
| `fa.identity` | `G` |
| `fa.center` | `H G H`, with `H = I − J/n`, in O(n²) |
| `fa.trace_normalize` | `G / tr G` (requires a positive trace) |
| `fa.frobenius_normalize` | `G / ‖G‖_F` |
| `fa.max_eig_normalize` | `G / λ_max` |

## Parametrised transforms

::: fannar.angular

::: fannar.shift

::: fannar.whiten

::: fannar.spectral

::: fannar.psd_project

::: fannar.transforms.hadamard_power

::: fannar.transforms.discarded_mass
