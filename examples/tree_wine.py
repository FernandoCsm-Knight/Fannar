"""Exemplo de árvore: uma árvore de decisão do scikit-learn sobre a base `wine`, com rótulos em texto.

    pip install "fannar[sklearn,pandas]"
    python tree_wine.py

Mostra `fannar.tree.TreeInterpreter`: um "layer" por profundidade, a folga medida em desvios-padrão
do treino (`X_train=`), rótulos em texto (mapeados para `tree.classes_`) e as mesmas explicações que
um modelo PyTorch tem. É a mesma tabela do `tabular_wine.py`, agora com outro modelo.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from sklearn.datasets import load_wine  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from fannar.tree import TreeInterpreter  # noqa: E402

out = Path("fannar_tree_wine")
out.mkdir(exist_ok=True)

data = load_wine(as_frame=True)
X = data.data
y = np.array(["barolo", "grignolino", "barbera"])[data.target]   # rótulos em texto
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
print(export_text(tree, feature_names=list(X.columns), max_depth=2))

# o piso: uma árvore com a mesma profundidade treinada com os rótulos embaralhados
shuffled = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, np.random.default_rng(0).permutation(y_tr))

interp = TreeInterpreter(tree, X_te, y_te, X_train=X_tr, reference_model=shuffled)
print("classes:", interp.class_names, "| camadas:", interp.layers)

report = interp.evaluate()
print(f"\nacurácia no teste: {report.meta['accuracy']:.3f}")
print(report.to_markdown())
interp.plot_report(report).savefig(out / "report.png", dpi=150)
interp.plot_embedding().savefig(out / "embedding.png", dpi=150)

names = interp.class_names
expl = interp.explain_pair(0)
print(f"\npar ({expl.i}, {expl.j}): {names[expl.pred_i]} x {names[expl.pred_j]}")
for feat, vi, vj, share in expl.top(3):
    print(f"  {feat:30s} i = {vi:7.2f}  j = {vj:7.2f}  parcela {share:.0%}")
print("parcelas por profundidade (atributo principal):",
      {layer: expl.top(1, layer)[0][0] for layer in expl.layers})

errs = interp.explain_errors()
print(f"\n{len(errs.errors)} erros:", [(names[t], names[p], n) for t, p, n in errs.confusions()])
for feat, share, first in errs.top(3):
    print(f"  {feat:30s} parcela nos erros {share:.0%}, primeiro em {first:.0%}")
errs.plot().savefig(out / "errors.png", dpi=150)

print("\nteste de trocas (área, maior = melhor):", interp.swap_test(20))
