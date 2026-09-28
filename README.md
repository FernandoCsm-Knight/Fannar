# Fannar

Espaços de representação de modelos treinados a partir de matrizes de Gram de várias fontes de
informação: um *kernel* positivamente definido por fonte, transformações admissíveis, composição
pelo produto de Hadamard e a geometria e a energia espectral da matriz resultante.

**Fannar** é o nome do método, e esta biblioteca é a sua implementação, a do artigo *Construção de Espaços via Tensores de Similaridade em
Espaços de Hilbert com Núcleo Reprodutor: Um Método de Interpretabilidade Mecanicista*.

```bash
pip install fannar               # núcleo: numpy + matplotlib
pip install "fannar[all]"        # + PyTorch, scikit-learn e pandas
pip install -e ".[test]"         # a partir deste diretório, para desenvolver
```

## A ideia em uma linha

```
G = T( T_1(K_1) ∘ T_2(K_2) ∘ … ∘ T_d(K_d) )
```

`K_l` é a matriz de Gram bruta da fonte `l` (de um *kernel* ou fornecida por você), `T_l` a
transformação da componente, `∘` o produto de Hadamard e `T` a transformação final. Pelo teorema
de Schur, `G` é semidefinida positiva para qualquer escolha, e por isso admite uma geometria
(distâncias, vizinhanças, conjuntos de nível) e uma decomposição espectral de energia.

A biblioteca separa, como o artigo, duas coisas:

* **o método** — a construção acima e as leituras sobre `G` (`KernelPipeline`, `GramSpace`);
* **a instanciação** — quais objetos, quais fontes, quais *kernels* e transformações
  (`TorchSources`, `TreeSources`, ou uma função sua).

## Início rápido: um modelo PyTorch

```python
import copy
import fannar as fa

initial = copy.deepcopy(model)        # guarde antes de treinar: é o piso das leituras
# ... treine `model` ...

interp = fa.Interpreter(
    model, X_test, y_test,            # o modelo treinado e as amostras a interpretar
    layers=["layer1", "layer2", "layer3"],   # nomes de model.named_modules()
    reference_model=initial,          # opcional: piso (a mesma rede na inicialização)
    feature_names=names,              # opcional: para as explicações de pares
)

report = interp.evaluate()            # leituras por camada: conjunta, referências e piso
print(report.to_markdown())           # também: report.to_frame(), report.to_json("r.json")

interp.plot_report(report)            # leituras × camadas
interp.plot_embedding()               # o espaço de cada camada nos seus dois modos dominantes
interp.plot_gram("layer2")            # componentes, produto de Hadamard e Gram final
interp.plot_spectrum("layer2")        # autovalores e posto de participação

expl = interp.explain_pair(0)         # por que a amostra 0 difere do vizinho mais próximo
expl.top(5)                           #   com outra predição? [(atributo, x_i, x_j, parcela)]
expl.plot()                           # parcela de cada atributo, camada a camada
interp.swap_test(50)                  # validação por trocas: método × |Δx| × aleatória

emap = interp.energy_map(image, "layer2")   # CNN: energia retida em cada posição da imagem
```

Por padrão, cada camada fornece três fontes, como no artigo:

| fonte | o que é |
|---|---|
| `activations` | a saída da camada, reduzida a uma grade comum de posições |
| `gradients` | a derivada da margem contrastiva `f_c − média(f_{c'≠c})` em relação a essa saída |
| `parameters` | os filtros que o objeto ativou, `p[:, s] = Σ_c A[c, s] W[c, :]` |

e o *pipeline* `"paper"`: *kernel* linear, `(T_a(K) + J)/2` em cada componente, produto de Hadamard
e `T_tr ∘ T_c`.

### Dois cuidados que o código impõe

* **Alvo do gradiente.** Com `target="predicted"`, o gradiente carrega a própria predição, e as
  leituras sobre a decisão ficam circulares. O padrão é `target="fixed"` (classe `fixed_class`
  para todos os objetos).
* **Piso.** Toda leitura precisa de uma referência. Passe `reference_model` (a mesma arquitetura
  com os pesos iniciais): o relatório mede a geometria dessa rede contra o comportamento da rede
  treinada, de modo que a única diferença entre a leitura e o seu piso é o treinamento.

## Sua própria matriz de Gram

Qualquer componente pode vir de um *kernel* seu, de uma transformação sua ou de uma matriz que
você já calculou:

```python
pipe = fa.KernelPipeline(
    {
        "estado":   fa.Component(fa.Linear(), fa.angular() >> fa.shift(1.0)),
        "texto":    fa.Component(fa.RBF(gamma=0.1), fa.identity),
        "meu":      fa.Component(lambda A, B: (A @ B.T + 1) ** 2),      # qualquer função k(X, Y)
    },
    final=fa.center >> fa.trace_normalize,
)
space = pipe(sources={"estado": S, "texto": T, "meu": M}, labels=y)
space = pipe(grams={"estado": K1, "texto": K2, "meu": K3})           # matrizes já prontas
space = fa.GramSpace(minha_matriz_psd, labels=y)                     # ou direto uma G sua
```

Transformações disponíveis (compõem com `>>`, da esquerda para a direita): `identity`, `center`,
`trace_normalize`, `frobenius_normalize`, `max_eig_normalize`, `angular()`, `shift(c)`,
`whiten()`, `spectral(f)`, `psd_project()`. *Kernels*: `Linear`, `Cosine`, `RBF`, `Polynomial`,
`FunctionKernel`. *Presets*: `fa.paper_pipeline()`, `fa.single_source("activations")`,
`fa.positions_pipeline()`.

> Com *kernels* gaussianos, o produto de Hadamard vira identicamente uma soma das métricas das
> fontes (o produto de gaussianas é a gaussiana da soma dos expoentes). Para preservar a interação
> multiplicativa entre as fontes, prefira `Linear`/`Cosine`.

## O espaço de representação (`GramSpace`)

```python
space.distances               # pseudométrica d(i,j)² = G_ii + G_jj − 2 G_ij
space.neighbors(i, k)         # k vizinhos mais próximos
space.level_band(i, r, delta) # faixa de nível Γ_{r,δ}(i)
space.equivalence_classes()   # objetos indistinguíveis para este espaço
space.participation_rank()    # (tr G)² / ‖G‖²_F, sem decomposição espectral
space.energies()              # E_∥, E_⊥ e E_ratio por objeto
space.containment(labels)     # direções de classe dentro de V_r (r = K − 1)
space.embedding(2)            # coordenadas nos dois modos dominantes (PCA de núcleo)
space.cka(outra)              # alinhamento centrado
space.decompose(K)            # G = a·K + R: o que K explica e o resíduo
```

## Outros modelos

```python
# árvore de decisão do scikit-learn: fontes caminho, folga e distribuição de classes por profundidade
interp = fa.Interpreter(tree, X_test, y_test)

# qualquer modelo: uma função que devolve {camada: {fonte: array}} e as predições
def fontes(X):
    return {"saida": {"proba": modelo.predict_proba(X), "folhas": modelo.apply(X)}}, modelo.predict(X)

interp = fa.Interpreter(modelo, X_test, y_test, extractor=fontes,
                        pipeline=fa.KernelPipeline(["proba", "folhas"]), baselines={})
```

## Leituras do relatório

| leitura | pergunta |
|---|---|
| `knn_prediction` | os vizinhos de um objeto têm a mesma predição que ele? |
| `correctness_auc` | a margem geométrica antecipa onde o modelo erra? |
| `confusion_agreement` | as classes que o modelo confunde estão próximas? (≥ 5 classes) |
| `containment` | as direções de classe estão no subespaço retido? |
| `e_ratio_error_auc` | a energia fora do subespaço retido sinaliza os erros? |

## Custo

Por matriz de Gram, `Θ(n²D + n³)` com a decomposição espectral completa (`n` objetos, `D` a
dimensão total das fontes); o posto de participação sai em `Θ(n²)`. A explicação de um par custa
uma passagem direta e uma reversa sobre `F + 2` linhas, linear no número de atributos `F`.

## Testes

```bash
pip install -e ".[test]" && pytest
```
