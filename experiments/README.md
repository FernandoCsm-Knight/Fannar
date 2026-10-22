# Espaço de representação por núcleos: bases tabulares, Oxford-IIIT Pet e CIFAR-100

Implementação do método de `docs/` (tensor de similaridade em RKHS, pipeline núcleo-transformação)
e dos testes que avaliam se o espaço de representação reflete o comportamento do modelo.

## Reproduzir tudo do zero

```bash
python -m venv ../.venv && ../.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
../.venv/Scripts/pip install -e "..[all]" scipy seaborn shap lime dice-ml pyarrow
bash run_tabular.sh ../.venv/Scripts/python          # dez bases, letter, sintético, árvore, figuras (~6 h)
bash run_pendigits.sh ../.venv/Scripts/python        # estudos de caso na pendigits (~3 h)
bash run_images.sh ../.venv/Scripts/python 8         # Pet (ResNet, ViT) e CIFAR-100, 8 sementes (~7 h)
python ablations_stats.py && python ablations_figure.py && python semantics_figure.py
python tree_case_figure.py --pairs 4 51              # pares típicos da semente 0
```

Medido numa RTX 4060 (8 GB). As figuras do artigo são copiadas de `outputs/` para `../docs/figs/`
(`res_estabilidade` ← `tabular_suite/suite_fixed`, `res_pets` ← `pets_suite/suite_fixed`, `res_cifar` ←
`cifar100_suite/suite_fixed`, `res_letter` ← `letter_case/case_study`, `res_pendigits` ←
`pendigits_case/case_study`, `res_orcamento` ← `letter_budget/budget`, `res_arvore` ← `tree_case/tree_case`,
`res_arvore_caminhos` ← `tree_case/tree_vs_method`, `res_campo_resnet`/`res_niveis` ← `pets_regions/`,
`res_campo_vit` ← `pets_regions_vit/`, `res_diferencas` ← `tabular_differences/differences`,
`res_complexidade` ← `complexity/complexity`, `disc_espaco` ← `discussion/scatter`,
`gram_construcao` ← `methodology/gram_construcao`; `res_ablacoes` e `res_semantica` já saem em `../docs/figs/`).

## Dados

| base | onde fica | o que é |
|---|---|---|
| 10 bases tabulares | OpenML pelo scikit-learn; se o OpenML não responder, os originais da UCI/StatLog em `~/Desktop/torch/data/tabular/` (`src/data_mirror.py`, mesma ordem de classes; `FANNAR_DATA=openml\|mirror` força um dos dois) | letter, pendigits, optdigits, satimage, segment, har, mfeat-factors, gas-drift, dna, covertype (subamostrada para 30 000); todas multiclasse |
| Titanic (OpenML 40945) | idem | só exemplo de leitura: 11 atributos com nome, binário, fora da suíte de estabilidade |
| Oxford-IIIT Pet | `~/Desktop/torch/data/oxford-iiit-pet` (imagens e anotações oficiais); cache em `outputs/pets_128.npz` | 37 raças, 7 349 imagens, máscara do animal em cada imagem (só verdade externa, nunca entra no treino); a espécie é a referência semântica de P9 |
| CIFAR-100 | `~/Desktop/torch/data/cifar100/cifar-100-python`; cache em `outputs/cifar100_32.npz` | 100 classes em 20 superclasses (referência semântica de P9, nunca entra no treino) |

## Scripts

| script | pergunta | saída |
|---|---|---|
| `tabular_suite.py` | o espaço de representação reflete a decisão, o erro e a confusão do modelo? como se compara a A, Γ, RSA e CKA? é estável entre bases e sementes? | `outputs/tabular_suite/` |
| `pets_suite.py` | o mesmo, por imagem, no Pet (ResNet), entre sementes | `outputs/pets_suite/` |
| `pets_regions.py` | o campo de energia e os conjuntos de nível caem no animal (máscara)? apontam o que a rede usa (deleção)? | `outputs/pets_regions/` |
| `tabular_differences.py` | por que esta amostra teve outra classificação? — quais **atributos** | `outputs/tabular_differences/` |
| `pets_differences.py` | por que esta imagem teve outra classificação? — quais **regiões**; consulta de região (`--grid`, `--query`) | `outputs/pets_differences/` |
| `titanic_example.py` | exemplo legível da mesma explicação, com atributos que têm nome (sexo, idade, classe) | `outputs/titanic_differences/exemplo.md` |
| `letter_case_study.py` | estudo de caso na `letter`: o método contra o SHAP (Shapley exato com j de referência e SHAP da biblioteca) e contra árvores (substituta e nos rótulos), par a par e em importância global | `outputs/letter_case/` |
| `tree_case_study.py` | o método aplicado a uma árvore de decisão (instanciação sem rede: fontes caminho, folga e classes por profundidade); recupera a explicação da própria árvore? | `outputs/tree_case/` |

### Ablações e estatística (Subseção `subsec:adicionais` do artigo)

| script | item | saída |
|---|---|---|
| `ablations.py` | (a) produto × soma (= concatenação) × sem T_a × sem J; (b) margem geométrica × softmax, margem dos logits, trust score e Mahalanobis; (c) k-vizinhos nos logits e nas ativações finais | `outputs/ablations/<família>/` |
| `synthetic_counterfactual.py` | (d) leitura de pares com verdade de base sintética, contra Shapley exato, gradientes integrados, LIME e DiCE (`--dice`) | `outputs/synthetic_counterfactual/` |
| `tree_case_study.py` | (e) o ranking `no_path` (folga ∘ classes) entra na tabela e na datação | `outputs/tree_case/` |
| `ablations_stats.py` | (f) datação sem tendência; (h) Wilcoxon + Holm, IC bootstrap, Friedman + Nemenyi — só lê os JSONs, não treina | `outputs/stats/` |
| `ablations_figure.py` | figura `res_ablacoes.pdf` | `../docs/figs/` |

Ordem de execução (dependências: `pip install -e "..[all]" scipy seaborn shap lime`, e `dice-ml` para o DiCE):

```bash
python ablations.py --family tabular --seeds 5           # horas: re-treina os MLPs
python ablations.py --family pets --arch resnet --seeds 3 # usa outputs/pets_models/ se existir
python ablations.py --family pets --arch vit --seeds 3
python synthetic_counterfactual.py --seeds 5 --pairs 100 --dice
python tree_case_study.py --seeds 5 --pairs 100 --out outputs/tree_case_v2   # os runs antigos não têm `no_path`
python ablations_stats.py --suite pets_vit=<pasta da suíte do transformador>
python ablations_figure.py
```

Mais sementes no Pet (item (h)) são só `--seeds N` em `pets_suite.py` e em `ablations.py`: com três, o menor p
possível do Wilcoxon é 0,25, e `ablations_stats.py` informa esse piso em vez de omitir o teste. O item (g) do
protocolo (modelos pré-treinados e outra base de imagens) não tem script: ficou como trabalho futuro no texto.

O modelo do Pet é treinado sob demanda por `src/train_pets.py` (`python -m src.train_pets --seed 0`) e
guardado em `outputs/pets_models/` junto com o estado inicial, que é o controle não treinado.

## Módulos (`src/`)

- **Método:** `kernels.py` (transformações admissíveis, Hadamard), `geometry.py` (componente angular deslocada,
  pseudodistância, RSA), `sources.py` (as três fontes e a Gram conjunta por objeto), `spectral.py` (energia
  espectral, V_r, Ê_ratio), `pixel_energy.py` (posições como objetos: campo de energia, deleção),
  `level_sets.py` (conjuntos de nível com distância angular), `cross_pair.py` (Gram cruzada de duas imagens).
- **Leituras e relatório:** `suite.py` (todas as leituras por objeto, pisos, afirmações, agregação),
  `behaviour.py`, `report.py`.
- **Modelos e dados:** `model.py` (ResNet), `mlp.py` (MLP residual com a mesma interface), `data_tabular.py`,
  `data_pets.py`, `train_pets.py`.
- **Figuras:** `theme.py` (identidade visual) e `plots_*.py`.

## Cuidados que o código impõe (todos vieram de erros medidos)

- **Tautologia de Γ.** Com a margem da classe predita, o gradiente carrega a predição (no MLP, 36% da variação
  de Γ no `stem` e 100% no `b6`). Toda execução roda também o **alvo fixo**, e as leituras referidas à predição
  só valem nessa variante.
- **Dimensão.** Contenção em V_r e Ê_ratio dependem de r, e a conjunta retém ~2,5× mais modos que A; entre
  representações, a comparação é feita com r fixo (K − 1, ou o r de A).
- **Confusão.** A contagem de erros na amostra deixa 80–96% dos pares de classes em zero em bases fáceis; a
  leitura principal é a **confusão suave** (probabilidade média, no teste inteiro).
- **Centro.** As lesões/animais ficam no meio da foto; toda leitura de localização tem um piso que só marca o centro.
- **Pisos.** O piso das leituras de comportamento é o **cruzado**: a mesma construção na rede não treinada (mesmos
  pesos iniciais) contra o comportamento da rede treinada.
- **Degenerescência.** Representações sem geometria (Γ no último bloco sob alvo fixo) ficam indefinidas, não viram ruído.

## Arquivo morto

`_arquivo/` guarda o código, as saídas e os modelos anteriores (CIFAR-10 e Cat/Dog), as rodadas antigas da suíte
tabular e a rodada do Pet com a receita de treino anterior (`*_v1`). O projeto não tem git; nada ali é usado pelo
código atual.

## Estado do texto

`paper/metodologia.tex` e `paper/resultados.tex` foram reescritos para estes experimentos. Os trechos que ainda
dependem de execuções pendentes (Oxford Pet com a receita atual, transformador, tabelas do apêndice e a figura
`gram_construcao.pdf`) estão marcados com `% PENDENTE` no próprio `.tex`.
