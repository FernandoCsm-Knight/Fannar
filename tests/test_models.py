import copy

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import fannar as fa  # noqa: E402

torch = pytest.importorskip("torch")
nn = torch.nn


def make_data(n=120, f=6, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = rng.standard_normal((n, f)).astype(np.float32)
    X[:, 0] += 2.0 * y
    X[:, 1] -= 1.5 * (y % 2)
    return X, y


def mlp(f=6, k=4):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(f, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, k))


def trained_mlp(X, y):
    model = mlp(X.shape[1], int(y.max() + 1))
    initial = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(150):
        opt.zero_grad(); nn.functional.cross_entropy(model(xt), yt).backward(); opt.step()
    return model.eval(), initial.eval()


def test_torch_sources_shapes_and_parameter_source():
    X, y = make_data()
    model, _ = trained_mlp(X, y)
    src = fa.TorchSources(model, ["1", "3"])(X)
    a, p = src.layers["1"]["activations"], src.layers["1"]["parameters"]
    assert a.shape == (len(X), 32, 1) and src.layers["1"]["gradients"].shape == a.shape
    w = model[0].weight.detach().double().numpy()
    np.testing.assert_allclose(p[:, :, 0], a[:, :, 0] @ w, atol=1e-6)
    assert src.probabilities.shape == (len(X), 4) and np.allclose(src.probabilities.sum(1), 1)


def test_fixed_vs_predicted_target():
    X, y = make_data()
    model, _ = trained_mlp(X, y)
    g_fixed = fa.TorchSources(model, ["3"], target="fixed", fixed_class=0)(X).layers["3"]["gradients"]
    g_pred = fa.TorchSources(model, ["3"], target="predicted")(X).layers["3"]["gradients"]
    assert not np.allclose(g_fixed, g_pred)


def test_interpreter_end_to_end_with_floor_and_plots():
    X, y = make_data()
    model, initial = trained_mlp(X, y)
    interp = fa.Interpreter(model, X, y, layers=["1", "3"], reference_model=initial,
                            feature_names=[f"f{t}" for t in range(6)])
    rep = interp.evaluate()
    assert set(rep.representations) == {"joint", "activations", "floor"}
    assert rep.series("knn_prediction").shape == (2,)
    assert "k-NN" in rep.to_markdown()
    assert rep.to_frame().shape[0] > 0
    for fig in (interp.plot_report(rep), interp.plot_embedding(), interp.plot_gram("3"), interp.plot_spectrum("3")):
        assert fig.axes
    expl = interp.explain_pair(0)
    assert expl.j is not None and set(expl.layers) == {"1", "3"}
    assert len(expl.top(3)) == 3 and expl.plot().axes
    res = interp.swap_test(10, layer="3")
    assert 0 <= res["method"] <= 1 and res["pairs"] == 10


def test_pair_scores_match_direct_distance():
    X, y = make_data()
    model, _ = trained_mlp(X, y)
    interp = fa.Interpreter(model, X, y, layers=["3"]).fit()
    expl = interp.explain_pair(0, 5)
    rows = fa.pairs.swap_rows(X[0].astype(float), X[5].astype(float))
    src = interp.extractor(rows.astype(np.float32))
    sp = fa.paper_pipeline()(sources=src.layers["3"])
    np.testing.assert_allclose(expl.scores["3"], sp.distances[0, 1] - sp.distances[0, 2:])


def test_cnn_energy_map_and_tokens():
    torch.manual_seed(0)
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3)).eval()
    X = np.random.default_rng(0).standard_normal((20, 3, 12, 12)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, 20)
    interp = fa.Interpreter(cnn, X, y, layers=["1", "3"], grid=2)
    assert interp.sources is None and interp.layers == ["1", "3"]
    assert interp.sources.layers["3"]["activations"].shape == (20, 8, 4)
    assert interp.sources.layers["3"]["parameters"].shape == (20, 72, 4)  # conv weight 8 x (8*3*3)
    emap = interp.energy_map(X[0], "3", max_side=6)
    assert emap.shape == (6, 6) and np.all(emap >= -1e-12)
    assert interp.plot_energy_map(X[0], "3", 6).axes

    class Tok(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(4, 8)
            self.mix = nn.Linear(8, 8)
            self.head = nn.Linear(8, 3)

        def forward(self, x):  # (N, L, 4)
            h = torch.relu(self.mix(torch.relu(self.embed(x))))
            return self.head(h.mean(1))

    tok = Tok().eval()
    Xt = np.random.default_rng(2).standard_normal((10, 5, 4)).astype(np.float32)
    src = fa.TorchSources(tok, ["mix"], layout="tokens", grid=None)(Xt)
    assert src.layers["mix"]["activations"].shape == (10, 8, 5)


def test_custom_extractor_and_own_grams():
    X, y = make_data(60)
    feats = np.hstack([X, X**2])

    def extractor(Z):
        Z = np.asarray(Z)
        return {"only": {"raw": Z, "sq": Z**2}}, (Z[:, 0] > 2).astype(int)

    interp = fa.Interpreter(object(), X, y, extractor=extractor,
                            pipeline=fa.KernelPipeline(["raw", "sq"]), baselines={})
    assert interp.layers == ["only"]
    K = feats @ feats.T
    sp = interp.space_from_grams({"mine": K})
    assert sp.n == 60 and sp.is_psd()
