# Matemática operacional da fannar

Este documento resume os objetos implementados e a correspondência com a
teoria (RKHS, produto tensorial, subespaço principal). Não repete as
demonstrações; foca no que o código calcula.

## 1. Kernel e matriz de Gram

Um kernel `k` produz `K_ij = k(x_i, x_j)`, simétrica e PSD quando válido. Os
objetos `x_i` são as linhas de `X` com shape `(n, d)`; `K` tem shape `(n, n)`.

## 2. Pipeline núcleo-transformação `(κ, T)`

`K = T(κ(X))`. As transformações `T` em `fannar.transforms` são equivariantes
por permutação dos objetos e preservam (quando possível) simetria e PSD:

- `Identity` — estrutura absoluta.
- `Centering` `HKH` — remove o centroide (`H = I − 11ᵀ/n`).
- `TraceNormalize` `K/tr(K)` — traço 1.
- `FrobeniusNormalize` `K/‖K‖_F`.
- `MaxEigenvalueNormalize` `K/λ₁`.
- `AngularNormalize` `K_ij/√(K_ii K_jj)` — produto interno cosseno.
- `SpectralWhitening` — potência espectral `V diag((λ+ε)^p) Vᵀ` (ou achatamento).
- `Compose` — composição (a classe é fechada por composição).

## 3. Produto tensorial via Hadamard

O produto tensorial de RKHS corresponde, na forma matricial, ao produto de
Hadamard das Grams componentes:

```
K_total = K_1 ∘ K_2 ∘ … ∘ K_d
```

PSD é preservado (teorema de Schur). Implementado em `hadamard_combine`; há
versão em domínio log (`hadamard_combine_log`) para muitos fatores positivos.

A forma padrão por camada é `K_layer = K_W ∘ K_A ∘ K_G` (pesos, ativações,
gradientes). O critério é de **similaridade conjunta**: a similaridade só é alta
se as três fontes concordam.

## 4. Distância no RKHS

```
d(i,j)² = K_ii + K_jj − 2 K_ij
```

`distance_matrix` retorna `D` simétrica, diagonal nula, satisfazendo a
desigualdade triangular (por ser métrica de Hilbert).

## 5. Subespaço principal e energia

Dado `K = V Λ Vᵀ`, o subespaço principal `V_r` retém os `r` primeiros modos até
um limiar `τ`. Duas energias:

- **operador** `Σλ²` (Hilbert–Schmidt; default);
- **traço** `Σλ` (KPCA).

Para um vetor `u`, decompõe-se `u = P∥u + P⊥u` e define-se

```
residual_ratio(u) = ‖P⊥u‖² / ‖u‖²  ∈ [0, 1].
```

Métricas: `euclidean` (razão de energia ortogonal direta) e `rkhs` (ponderação
por `1/λ_k`, base `{√λ_k φ_k}`).

## 6. Coordenadas espectrais e fidelidade

```
z_i = (√λ₁ V_i1, …, √λ_r V_ir)     # MDS clássico
Fid(r) = Σ_{k≤r} λ_k / Σ_k λ_k     # fidelidade da projeção (traço)
```

`Fid` (traço) é distinta de `Cov` (energia do operador): a primeira mede
preservação da soma de distâncias na projeção; a segunda, fração de energia do
operador no subespaço.

## 7. Diagnósticos e perfil explicativo

- `dispersion` = média de `D`.
- `stability(i)` = média das distâncias aos `k` vizinhos.
- `coverage(τ)` = `r(τ)/C`.
- `residual_energy(u, τ)` = razão de energia ortogonal.
- `build_laplacian` = afinidade gaussiana `A_ij = exp(−D_ij²/σ²)` e laplaciano
  normalizado `L = I − Deg^{-1/2} A Deg^{-1/2}`.
- `explanatory_profile` = vetor de 8 eixos `(DN, IS, RE, Pr, Un, CM, NM, In)`,
  cada um uma fração genuína em `[0,1]`; eixos indefinidos retornam
  `available=False`.
