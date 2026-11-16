# Cost

With $n$ objects, $D$ the total dimension of the sources and $F$ the number of input features:

| step | cost |
|---|---|
| one Gram matrix (kernels, transforms, Hadamard product) | $\Theta(n^2 D)$ |
| distances, neighbours, centring, participation rank | $\Theta(n^2)$ |
| full spectral decomposition (energies, embedding, containment) | $\Theta(n^3)$, computed once per space |
| a pair explanation | one forward and one backward pass over $F + 2$ rows |
| a swap-test curve | one forward pass over $F + 1$ rows |

The spectral transforms (`whiten`, `spectral`, `psd_project`, `max_eig_normalize`) also cost
$\Theta(n^3)$; the others are $O(n^2)$.

## Keeping it fast

- **Number of objects.** Memory and time grow with $n^2$ and $n^3$. A few thousand objects per
  space is comfortable on a laptop; subsample `X` beyond that.
- **Grid.** For convolutional and token layers, `grid` (default 4) sets how many positions are
  kept per axis, so $D$ grows with `C × grid²`. Lower it for wide layers.
- **Sources.** `sources=("activations", "gradients")` skips the parameter source, the largest
  for layers with many inputs per channel.
- **Device and batch size.** Extraction runs on the model's device in batches of
  `batch_size=256`; the Gram algebra runs on the CPU in float64.
- **Errors.** `explain_errors(max_errors=..., reference=...)` caps how many pairs are explained.
