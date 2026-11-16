"""The domain modules (tabular, image, text, tree) are thin layers over Interpreter: same numbers."""

import subprocess
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import fannar as fa  # noqa: E402


def same_report(a, b):
    assert a.layers == b.layers and set(a.values) == set(b.values)
    for rep in a.values:
        for layer in a.values[rep]:
            for key, v in a.values[rep][layer].items():
                np.testing.assert_allclose(v, b.values[rep][layer][key], equal_nan=True)


def test_domain_modules_load_on_demand():
    code = ("import sys, fannar as fa; "
            "assert not any(f'fannar.{m}' in sys.modules for m in ('tabular', 'image', 'text', 'tree')); "
            "fa.tabular, fa.image, fa.text, fa.tree; "
            "assert fa.tree.TreeInterpreter and fa.tabular.TabularInterpreter; print(fa.__version__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert {"tabular", "image", "text", "tree"} <= set(dir(fa))
    with pytest.raises(AttributeError):
        fa.nope


# ---------------------------------------------------------------- tree
sk = pytest.importorskip("sklearn.tree")


def tree_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (400, 5))
    y = (X[:, 2] > 5).astype(int) + 2 * (X[:, 4] > 5).astype(int)
    return X, y


def test_tree_interpreter_matches_interpreter():
    X, y = tree_data()
    tree = sk.DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    same_report(fa.tree.TreeInterpreter(tree, X[:120], y[:120]).evaluate(),
                fa.Interpreter(tree, X[:120], y[:120]).evaluate())


def test_tree_interpreter_scale_names_and_string_labels():
    pd = pytest.importorskip("pandas")
    X, y = tree_data()
    names = np.array(["low", "mid", "high", "top"])[y]
    cols = [f"col{t}" for t in range(5)]
    df = pd.DataFrame(X * [1, 1, 1, 1, 100], columns=cols)
    tree = sk.DecisionTreeClassifier(max_depth=4, random_state=0).fit(df.to_numpy(), names)
    interp = fa.tree.TreeInterpreter(tree, df[:150], names[:150], X_train=df)
    assert interp.feature_names == cols and interp.class_names == list(tree.classes_)
    assert np.array_equal(interp.y, np.searchsorted(tree.classes_, names[:150]))
    assert np.array_equal(tree.classes_[interp.predictions], tree.predict(df[:150].to_numpy()))
    np.testing.assert_allclose(interp.extractor.scale, df.to_numpy().std(0))
    assert "path" in interp.evaluate().representations
    assert interp.explain_pair(0).feature_names == cols
    assert interp.swap_test(5)["pairs"] == 5


# ---------------------------------------------------------------- torch
torch = pytest.importorskip("torch")
nn = torch.nn


def test_tabular_interpreter_matches_and_reads_dataframes():
    pd = pytest.importorskip("pandas")
    from test_models import make_data, trained_mlp
    X, y = make_data()
    model, initial = trained_mlp(X, y)
    same_report(fa.tabular.TabularInterpreter(model, X, y, layers=["1", "3"], reference_model=initial).evaluate(),
                fa.Interpreter(model, X, y, layers=["1", "3"], reference_model=initial).evaluate())

    cols = [f"f{t}" for t in range(X.shape[1])]
    labels = pd.Series(pd.Categorical.from_codes(y, ["a", "b", "c", "d"]))
    interp = fa.tabular.TabularInterpreter(model, pd.DataFrame(X.astype(np.float64), columns=cols), labels,
                                           layers=["3"])
    assert interp.feature_names == cols and interp.class_names == ["a", "b", "c", "d"]
    assert interp.X.dtype == np.float32 and np.array_equal(interp.y, y)
    assert interp.explain_pair(0).feature_names == cols

    strings = fa.tabular.encode_labels(np.array(["b", "a", "b"]))
    assert strings[0].tolist() == [1, 0, 1] and strings[1] == ["a", "b"]


def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3)).eval()


def test_image_interpreter_defaults_and_energy_map():
    cnn = small_cnn()
    X = np.random.default_rng(0).standard_normal((20, 3, 12, 12)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, 20)
    interp = fa.image.ImageInterpreter(cnn, X, y, layers=["1", "3"], class_names=["x", "y", "z"])
    assert interp.sources is None and interp.layers == ["1", "3"]
    assert interp.sources.layers["3"]["activations"].shape == (20, 8, 16)   # grid=4 -> 16 positions
    same_report(interp.evaluate(), fa.Interpreter(cnn, X, y, layers=["1", "3"], grid=4).evaluate())
    assert interp.energy_map(X[0], "3", max_side=6).shape == (6, 6)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                                         batch_size=7)
    from_loader = fa.image.ImageInterpreter(cnn, loader, y, layers=["3"]).fit()
    np.testing.assert_allclose(from_loader.sources.layers["3"]["activations"],
                               interp.sources.layers["3"]["activations"], atol=1e-6)


class Out:  # what a Hugging Face model returns: an object with .logits
    def __init__(self, logits):
        self.logits = logits


class FakeHF(nn.Module):
    """Embedding + a ModuleList of blocks that return tuples, called as model(input_ids=, attention_mask=)."""

    def __init__(self, vocab=20, d=8, k=3, n_blocks=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.ReLU()) for _ in range(n_blocks))
        self.head = nn.Linear(d, k)
        self.seen_mask = None

    def forward(self, input_ids, attention_mask=None):
        self.seen_mask = attention_mask
        h = self.embed(input_ids)
        for block in self.encoder.layer:
            h = block(h)
        m = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return Out(self.head(pooled))


def test_text_interpreter_with_wrapped_hf_model():
    torch.manual_seed(0)
    hf = FakeHF().eval()
    model = fa.text.wrap_hf(hf, pad_token_id=0)
    ids = torch.randint(1, 20, (24, 10))
    ids[:, 8:] = 0                                              # padding
    y = np.random.default_rng(0).integers(0, 3, 24)
    assert fa.text.block_names(model) == [f"model.encoder.layer.{t}" for t in range(3)]
    interp = fa.text.TextInterpreter(model, ids, y, grid=5)
    assert interp.layers == fa.text.block_names(model)
    assert interp.sources.layers["model.encoder.layer.2"]["activations"].shape == (24, 8, 5)
    assert hf.seen_mask is not None and int(hf.seen_mask[0, 9]) == 0
    same_report(interp.evaluate(), fa.Interpreter(model, ids, y, layers=fa.text.block_names(model), layout="tokens",
                                                  grid=5).evaluate())
    assert interp.energy_map(ids[0], "model.encoder.layer.1").shape == (10,)   # positions = tokens

    with pytest.raises(ValueError):
        fa.text.block_names(nn.Linear(2, 2))
