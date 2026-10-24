# Experimentos do artigo: bases tabulares, Oxford-IIIT Pet, CIFAR-100 e estudos de caso

Implementação da instanciação do método de `docs/` (tensor de similaridade em RKHS, pipeline
núcleo-transformação) e dos testes que avaliam se o espaço de representação reflete o comportamento do
modelo. As perguntas P1 a P8 são as da Subseção `subsec:perguntas` do artigo.

## Reproduzir tudo do zero

```bash
python -m venv ../.venv && ../.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
../.venv/Scripts/pip install -e "..[all]" scipy seaborn shap lime dice-ml pyarrow
bash run_tabular.sh ../.venv/Scripts/python          # vinte bases, sintético, figuras do método (~5 h)
bash run_images.sh ../.venv/Scripts/python 8         # Pet (ResNet, ViT) e CIFAR-100, 8 sementes (~7 h)
bash run_case.sh letter ../.venv/Scripts/python      # estudo de caso, decisão nítida (~3 h)
bash run_case.sh heloc ../.venv/Scripts/python       # estudo de caso, decisão fraca (~3 h)
python ablations_stats.py && python ablations_figure.py && python semantics_figure.py
python results_figures.py
FANNAR_CASE=letter python tree_case_figure.py --pairs <i> <j>   # dois pares típicos da semente 0
```

Medido numa RTX 4060 (8 GB). As bases tabulares são baixadas do OpenML pelo scikit-learn na primeira
execução. As figuras do artigo são copiadas de `outputs/` para `../docs/figs/` (`res_complexidade` ←
`complexity/complexity`, `res_arvore_caminhos` ← `tree_case_<base>/tree_vs_method`, `res_campo_resnet` e
`res_niveis` ← `pets_regions/`, `disc_espaco` ← `discussion/scatter`, `gram_construcao` ←
`methodology/gram_construcao`). As figuras compostas já saem em `../docs/figs/`: `res_comportamento`
(`results_figures.py`), `res_ablacoes` (`ablations_figure.py`) e `res_semantica` (`semantics_figure.py`).

## Dados

| base | onde fica | o que é |
|---|---|---|
| 20 bases tabulares | OpenML (`src/data_tabular.py`) | todas as multiclasse do OpenML-CC18 com ao menos 2 000 amostras, menos as duplicatas e a `CIFAR_10`, mais `gas-drift` e `covertype`; as maiores, subamostradas para 30 000 |
| HELOC (OpenML 46932) | OpenML | estudo de caso de decisão fraca: 23 atributos do histórico de crédito, binário; a `letter`, da suíte, é o de decisão nítida |
| Oxford-IIIT Pet | `~/Desktop/torch/data/oxford-iiit-pet`; cache em `outputs/pets_128.npz` | 37 raças, máscara do animal (só verdade externa); a espécie é a referência semântica de P8 |
| CIFAR-100 | `~/Desktop/torch/data/cifar100/cifar-100-python`; cache em `outputs/cifar100_32.npz` | 100 classes em 20 superclasses (referência semântica de P8) |

## Scripts

| script | pergunta | saída |
|---|---|---|
| `tabular_suite.py` | P1–P4 e P7 nas vinte bases: o espaço reflete a decisão, o erro e a confusão? como se compara a A, Ω, RSA e CKA? | `outputs/tabular_suite/` |
| `pets_suite.py` | o mesmo por imagem, no Pet (ResNet e ViT) e no CIFAR-100, mais P8 | `outputs/pets_suite*/`, `outputs/cifar100_suite/` |
| `ablations.py` | P2: produto × soma (= concatenação) × sem T_a × sem J; P1: margem geométrica × sinais de confiança | `outputs/ablations/<família>/` |
| `pets_regions.py` | P5: o campo de energia e os conjuntos de nível caem no animal? como se relacionam à deleção? | `outputs/pets_regions*/` |
| `tabular_differences.py` | P6 nas vinte bases: teste de trocas contra \|Δx\|, grad×Δx e gradientes integrados | `outputs/tabular_differences/` |
| `pets_differences.py` | P6 nas imagens: quais regiões distinguem i de j | `outputs/pets_differences/` |
| `synthetic_counterfactual.py` | P6 com verdade de base sintética | `outputs/synthetic_counterfactual/` |
| `case_study.py` | estudo de caso de P6 (`FANNAR_CASE=letter\|heloc`): rankings contra SHAP, LIME, Shapley exato, gradientes integrados e DiCE; construção do contrafactual; diagnóstico de Ω | `outputs/<base>_case/` |
| `case_budget.py` | P6: o método contra o Shapley amostrado e o LIME com o mesmo orçamento | `outputs/<base>_budget/` |
| `tree_case_study.py` | P6: o método aplicado a uma árvore de decisão (fontes caminho, folga e classes); recupera o nó de separação e a profundidade (P7)? | `outputs/tree_case_<base>/` |
| `tree_counterfactual.py` | P6: o contrafactual na árvore, contra o mínimo exato | `outputs/tree_counterfactual_<base>/` |
| `heloc_diagnosis.py` | por que a leitura de pares falha no perceptron do HELOC | `outputs/heloc_diagnosis/` |
| `ablations_stats.py` | datação sem tendência (P7), Wilcoxon + Holm, IC bootstrap, Friedman + Nemenyi; só lê os JSONs | `outputs/stats/` |
| `complexity_bench.py` | custo medido das etapas do método e das técnicas de P6 | `outputs/complexity/` |

Os modelos do Pet e do CIFAR-100 são treinados sob demanda por `src/train_pets.py` e guardados em
`outputs/*_models/` junto com o estado inicial, que é o controle não treinado. Todos os scripts retomam a
partir dos `runs/*.json` já gravados.

## Módulos (`src/`)

- **Método:** `kernels.py` (transformações admissíveis, Hadamard), `geometry.py` (componente angular
  deslocada, pseudodistância, RSA), `sources.py` (as três fontes e a Gram conjunta por objeto), `spectral.py`
  (energia espectral, V_r, Ê_ratio), `pixel_energy.py` (posições como objetos: campo de energia, deleção),
  `level_sets.py` (conjuntos de nível), `cross_pair.py` (Gram cruzada de duas imagens).
- **Leituras e relatório:** `suite.py` (leituras por objeto, pisos, afirmações, agregação), `behaviour.py`,
  `ablation.py`, `report.py`.
- **Explicações de referência:** `attributions.py` (Shapley exato e amostrado, SHAP, LIME, gradientes
  integrados, DiCE, contrafactual mínimo), `case.py` (bases, regra de pares e cálculos exatos na árvore dos
  estudos de caso).
- **Modelos e dados:** `model.py` (ResNet), `vit.py` (transformador), `mlp.py` (MLP residual), `data_tabular.py`,
  `data_pets.py`, `data_cifar.py`, `data_images.py`, `train_pets.py`.
- **Figuras:** `theme.py` (identidade visual) e `plots_*.py`.

## Cuidados que o código impõe (todos vieram de erros medidos)

- **Tautologia de Ω.** Com a margem da classe predita, o gradiente carrega a predição (no MLP, 100% da
  variação de Ω no `b6`). Toda execução roda também o **alvo fixo**, e as leituras referidas à predição só valem
  nessa variante.
- **Dimensão.** Contenção em V_r e Ê_ratio dependem de r, e a conjunta retém mais modos que A; entre
  representações, a comparação é feita com r fixo (Q − 1, ou o r de A).
- **Confusão.** A contagem de erros na amostra deixa a maior parte dos pares de classes em zero em bases fáceis;
  a leitura principal é a **confusão suave** (probabilidade média, no teste inteiro).
- **Centro.** Os animais ficam no meio da foto; toda leitura de localização tem um piso que só marca o centro.
- **Pisos.** O piso das leituras de comportamento é o **cruzado**: a mesma construção na rede não treinada
  (mesmos pesos iniciais) contra o comportamento da rede treinada.
- **Degenerescência.** Representações sem geometria (Ω no último bloco sob alvo fixo) ficam indefinidas, não
  viram ruído.
- **Paralelismo.** `run_case.sh` roda as sementes em processos paralelos com poucas *threads* cada; sem esse
  limite, os processos disputam os mesmos núcleos e ficam muitas vezes mais lentos.
