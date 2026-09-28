import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import fannar as fa  # noqa: E402

sk = pytest.importorskip("sklearn.tree")


def test_tree_interpreter_recovers_the_split():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (400, 5))
    y = (X[:, 2] > 5).astype(int) + 2 * (X[:, 4] > 5).astype(int)  # 4 classes from features 2 and 4
    tree = sk.DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    interp = fa.Interpreter(tree, X[:120], y[:120])
    rep = interp.evaluate()
    assert "path" in rep.representations and len(rep.layers) == tree.get_depth()
    # a pair that differs only in the decisive feature 2 (and one irrelevant feature)
    a = np.array([[3.0, 1, 2.0, 7, 2.0]]); b = a.copy(); b[0, 2] = 8.0; b[0, 0] = 1.0
    Z = np.vstack([X[:120], a, b])
    interp2 = fa.Interpreter(tree, Z, tree.predict(Z))
    expl = interp2.explain_pair(120, 121)
    assert expl.ranking()[0] == 2
    assert interp2.plot_embedding(include_baselines=False).axes
