# Fannar


Interpretabilidade geométrica de modelos de IA via **espaços de Hilbert com
núcleo reprodutor (RKHS)**, **produto tensorial de kernels** (produto de
Hadamard de matrizes de Gram), **decomposição espectral**, **energia induzida
por Gram** e **subconceitos por curvas de nível**.

A biblioteca formaliza a ideia de que cada camada de um modelo organiza seus
canais em um espaço de tipos cujo
kernel combinado é `K = K_W ∘ K_A ∘ K_G` (pesos ∘ ativações ∘ gradientes), e
extrai diagnósticos geométricos desse espaço.

## Instalação

```bash
pip install -e .            # núcleo (torch, numpy)
pip install -e ".[viz]"     # + matplotlib / plotly
pip install -e ".[dev]"     # + pytest, ruff, mypy, hypothesis
```

Requer Python ≥ 3.10 e PyTorch ≥ 2.0.

## Matemática operacional (resumo)

| Conceito | Implementação |
|---|---|
| Kernel `k(x,x')` → Gram `K` | `fannar.kernels` (`Linear`, `Cosine`, `RBF`, `Polynomial`) |
| Transformação pós-kernel `T(K)` | `fannar.transforms` (centramento, traço, Frobenius, angular, whitening, composição) |
| Produto tensorial `K₁∘…∘K_d` | `fannar.hadamard_combine` / `TensorGram` |
| Distância no RKHS `d²=Kᵢᵢ+Kⱼⱼ−2Kᵢⱼ` | `fannar.distance_matrix` |
| Subespaço principal `V_r` (energia τ) | `fannar.principal_subspace` |
| Energia capturada/residual | `fannar.energy_decomposition` |
| Energia induzida espacial `E_G(s)` | `fannar.concepts.gram_metric_usage`, `gram_induced_energy_surface` |
| Patches mais ativados por métrica | `fannar.concepts.top_metric_patches` |
| Coordenadas espectrais / fidelidade | `fannar.spectral_coordinates`, `embedding_fidelity` |
| Diagnósticos | `dispersion`, `stability`, `coverage`, `residual_energy`, `build_laplacian` |
| Perfil explicativo (8 eixos) | `fannar.explanatory_profile` |

Duas noções de energia, conforme a teoria:
- **operador** `Σλ²` (norma de Hilbert–Schmidt; ponderação quadrática dos modos);
- **traço** `Σλ` (convenção tipo KPCA).

A escolha é parâmetro (`energy="operator"` por padrão). Para **fidelidade de
visualização** usa-se a cobertura por traço (`embedding_fidelity`), que é uma
quantidade distinta da cobertura por energia do operador.

## Uso rápido

```python
import torch, fannar as f

X = torch.randn(32, 64)                 # 32 canais, 64 features
K_W = f.LinearKernel()(X)
K_A = f.CosineKernel()(X)
K_G = f.LinearKernel()(torch.randn(32, 64))

K = f.hadamard_combine([K_W, K_A, K_G]) # K_total
D = f.distance_matrix(K)
sub = f.principal_subspace(K, tau=0.9)

u = torch.randn(32)
ed = f.energy_decomposition(u, sub)
print(float(ed.residual_ratio))         # quanto de u ficou fora de V_r
```

### Pipeline por camada (pesos ∘ ativações ∘ gradientes)

```python
from fannar.pipelines import default_layer_pipeline, LayerRepresentations

reps = LayerRepresentations(weights=W, activations=A, gradients=G)  # cada (C, d)
geo = default_layer_pipeline().analyze(reps, tau=0.95, gradient_direction=g)
print(geo.coverage.detail, geo.profile.as_dict())
```

### Modelo PyTorch real

```python
from fannar.models import FannarExtractor
from fannar.pipelines import default_layer_pipeline

extractor = FannarExtractor(model, layers=["features.10"], capture=["weights","activations","gradients"])
reps = extractor.extract(inputs=batch, target=target_class, contrastive_class=None)
geo = default_layer_pipeline().analyze(reps["features.10"], gradient_direction=None)
```

### Visualização

```python
from fannar.visualization import plot_spectral_projection, plot_explanatory_radar
plot_spectral_projection(geo.K_total, dim=2, labels=geo.labels)
plot_explanatory_radar(geo.profile)
```

## O que é sólido e o que tem ressalvas

**Sólido e testado**: kernels,
transformações, produto de Hadamard, distância, decomposição espectral,
subespaço principal, energia residual, coordenadas espectrais, diagnósticos e
perfil explicativo, energia induzida por Gram e seleção de patches por métrica.
Propriedades verificadas: PSD preservado, traço unitário, diagonal de distância
nula, reconstrução espectral, `residual_ratio ∈ [0,1]`, `paralela + ortogonal =
u`, hooks removidos após extração e retorno consistente de patches ativados.

**Com ressalvas declaradas:**
- **Extração de modelos** é genérica para módulos nomeados padrão (Conv2d,
  Linear, blocos que retornam um tensor). O reshape "canal" segue convenções
  documentadas em `models/pytorch_adapter.py`; arquiteturas exóticas podem
  precisar de adaptador.
- **Métrica RKHS** em `energy_decomposition(metric="rkhs")` adota a base
  ortonormal `{√λ_k φ_k}` (ponderação por `1/λ_k`). O default `"euclidean"` é a
  razão de energia ortogonal direta — use-o salvo se souber que quer a métrica
  RKHS.
- **`hadamard_combine_log`** (domínio log) só é bem-definido para entradas
  positivas; para Grams com valores negativos use `hadamard_combine`.
- **`explanatory_profile`**: `Pr` exige estatísticas inter-contexto (rótulos);
  `Un` exige ≥ 2 camadas; `IS`/`RE`/`In` exigem vetor de gradiente. Eixos
  ausentes retornam `available=False` (nunca `NaN`).

## Estrutura

```
src/fannar/
  kernels/ transforms/ gram/        # núcleo matemático
  diagnostics/                      # dispersão, estabilidade, cobertura, perfil
  pipelines/                        # kernel / layer / model
  models/                           # extração PyTorch (hooks, adapters)
  concepts/                         # energia induzida, delta de Gram, patches por métrica
  visualization/ io/ utils/
```

## Licença

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
