import numpy as np
import pytest

import fannar as fa
from fannar import readings as R
from fannar.transforms import discarded_mass


def rand_psd(n=12, d=5, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d))
    return x @ x.T


# --------------------------------------------------------------------- kernels
def test_linear_and_cosine():
    x = np.random.default_rng(1).standard_normal((7, 3))
    np.testing.assert_allclose(fa.Linear()(x), x @ x.T)
    k = fa.Cosine()(x)
    np.testing.assert_allclose(np.diag(k), 1.0)
    assert np.all(np.abs(k) <= 1 + 1e-12)


def test_kernels_are_psd_and_named():
    x = np.random.default_rng(2).standard_normal((20, 4))
    for name in ("linear", "cosine", "rbf", "polynomial"):
        k = fa.kernels.as_kernel(name)(x)
        assert np.linalg.eigvalsh(k).min() > -1e-8 * np.trace(k)
    k = fa.kernels.as_kernel(lambda a, b: a @ b.T)(x)
    np.testing.assert_allclose(k, x @ x.T)
    with pytest.raises(ValueError):
        fa.kernels.as_kernel("nope")


def test_inputs_are_flattened():
    x = np.random.default_rng(3).standard_normal((5, 2, 3))
    np.testing.assert_allclose(fa.Linear()(x), x.reshape(5, -1) @ x.reshape(5, -1).T)


# --------------------------------------------------------------------- transforms
def test_center_equals_HGH():
    G = rand_psd()
    n = len(G)
    H = np.eye(n) - 1.0 / n
    np.testing.assert_allclose(fa.center(G), H @ G @ H, atol=1e-10)


def test_basic_properties():
    G = rand_psd()
    assert np.isclose(np.trace(fa.trace_normalize(G)), 1.0)
    np.testing.assert_allclose(np.diag(fa.angular()(G)), 1.0)
    S = (fa.angular() >> fa.shift(1.0))(G)
    assert S.min() >= -1e-12 and S.max() <= 1 + 1e-12
    W = fa.whiten()(G)
    np.testing.assert_allclose(W @ W, W, atol=1e-8)  # projector


@pytest.mark.parametrize("t", [fa.identity, fa.center, fa.trace_normalize, fa.frobenius_normalize,
                               fa.max_eig_normalize, fa.angular(), fa.shift(1.0), fa.whiten(),
                               fa.center >> fa.trace_normalize, fa.angular() >> fa.shift(0.5)])
def test_transforms_are_admissible(t):
    G = rand_psd(10, 4, seed=4)
    P = np.eye(10)[np.random.default_rng(5).permutation(10)]
    out = t(G)
    np.testing.assert_allclose(t(P @ G @ P.T), P @ out @ P.T, atol=1e-9)   # equivariance
    assert np.linalg.eigvalsh(out).min() > -1e-8 * max(abs(np.trace(out)), 1)  # PSD


def test_composition_order():
    G = rand_psd()
    np.testing.assert_allclose((fa.center >> fa.trace_normalize)(G), fa.trace_normalize(fa.center(G)))


# --------------------------------------------------------------------- pipeline
def test_hadamard_of_linear_components_is_the_tensor_product_gram():
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((9, 3)), rng.standard_normal((9, 4))
    pipe = fa.KernelPipeline({"a": fa.Component("linear", fa.identity), "b": fa.Component("linear", fa.identity)},
                             final=fa.identity)
    tensor = np.einsum("ni,nj->nij", a, b).reshape(9, -1)
    np.testing.assert_allclose(pipe(sources={"a": a, "b": b}).G, tensor @ tensor.T, atol=1e-10)


def test_user_grams_override_and_psd():
    rng = np.random.default_rng(7)
    feats = {s: rng.standard_normal((15, 6)) for s in fa.pipeline.PAPER_SOURCES}
    pipe = fa.paper_pipeline()
    sp1 = pipe(sources=feats)
    raw = {s: f @ f.T for s, f in feats.items()}
    sp2 = pipe(grams=raw)
    np.testing.assert_allclose(sp1.G, sp2.G, atol=1e-12)
    assert sp1.is_psd() and np.isclose(np.trace(sp1.G), 1.0)
    with pytest.raises(KeyError):
        pipe(sources={"activations": feats["activations"]})


def test_root_variant_reports_discarded_mass():
    rng = np.random.default_rng(8)
    feats = {s: rng.standard_normal((15, 6)) for s in ("a", "b", "c")}
    sp = fa.KernelPipeline(["a", "b", "c"], root=True)(sources=feats)
    assert 0.0 <= sp.info["root_discarded"] < 1.0 and sp.is_psd()


# --------------------------------------------------------------------- space
def test_pseudometric():
    sp = fa.GramSpace(rand_psd(15, 3))
    d = sp.distances
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    i, j, k = np.meshgrid(*[np.arange(15)] * 3, indexing="ij")
    assert np.all(d[i, k] <= d[i, j] + d[j, k] + 1e-9)


def test_equivalence_and_level_sets():
    x = np.array([[1.0, 0], [1.0, 0], [0, 1.0], [2.0, 0]])
    sp = fa.GramSpace(x @ x.T)
    groups = sp.equivalence_classes()
    assert any(set(g) == {0, 1} for g in groups)
    assert set(sp.level_set(0, 0.0)) == {0, 1}
    np.testing.assert_array_equal(sp.neighbors(0, 1), [1])


def test_energy_identities():
    sp = fa.GramSpace(fa.center(rand_psd(20, 6)))
    e = sp.energies()
    np.testing.assert_allclose(e["E_par"] + e["E_perp"], np.diag(sp.G), atol=1e-10)
    np.testing.assert_allclose(sp.mode_energy().sum(1), np.diag(sp.G), atol=1e-10)
    lam = sp.eigenvalues
    assert sp.participation_rank() == max(1, int(round(lam.sum() ** 2 / (lam**2).sum())))


def test_embedding_reproduces_G_with_all_modes():
    sp = fa.GramSpace(rand_psd(8, 8))
    xy, frac = sp.embedding(8)
    np.testing.assert_allclose(xy @ xy.T, sp.G, atol=1e-8)
    assert np.isclose(frac, 1.0)


def test_containment_and_cka():
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(3), 10)
    x = np.eye(3)[labels] * 5 + 0.1 * rng.standard_normal((30, 3))
    sp = fa.GramSpace(fa.center(x @ x.T), labels=labels)
    assert sp.containment() > 0.95
    assert np.isclose(sp.cka(sp), 1.0)
    dec = sp.decompose(sp.G)
    assert np.isclose(dec["alignment"], 1.0) and dec["discarded"] < 1e-8


# --------------------------------------------------------------------- readings
def test_auc_and_readings_on_separated_classes():
    assert R.auc([0.1, 0.2, 0.9, 0.8], [False, False, True, True]) == 1.0
    assert R.auc([1, 1, 1], [True, False, True]) == 0.5
    rng = np.random.default_rng(10)
    labels = np.repeat(np.arange(5), 12)
    x = np.eye(5)[labels] * 4 + 0.3 * rng.standard_normal((60, 5))
    sp = fa.GramSpace(fa.center(x @ x.T), labels=labels)
    assert R.knn_prediction(sp, labels) == 1.0
    assert np.all(R.geometric_margin(sp, labels) > 0)
    preds = labels.copy(); preds[:6] = (preds[:6] + 1) % 5
    out = fa.evaluate(sp, labels, preds)
    assert set(R.READINGS) <= set(out)
    assert np.isnan(R.confusion_agreement(fa.GramSpace(np.eye(4)), [0, 1, 2, 3]))
