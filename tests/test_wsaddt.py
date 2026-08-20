"""Smoke + determinism + legacy-equivalence tests for WSAD-DT."""
import sys, pathlib
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from wsad_dt import WSADDT

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _weak_data(seed=0, n_normal=200, n_anom=20, d=8):
    X, _ = make_blobs(n_samples=n_normal, centers=2, n_features=d,
                      cluster_std=0.5, random_state=seed)
    rng = np.random.RandomState(seed)
    A = rng.uniform(X.min(0) - 4, X.max(0) + 4, size=(n_anom, d))
    Xall = MinMaxScaler().fit_transform(np.vstack([X, A]))
    y_true = np.r_[np.zeros(n_normal), np.ones(n_anom)].astype(int)
    # weak labels: only 5 anomalies labeled, rest 0
    y_weak = np.zeros_like(y_true)
    labeled = rng.choice(np.where(y_true == 1)[0], 5, replace=False)
    y_weak[labeled] = 1
    return Xall, y_true, y_weak


def test_smoke_auc_and_api():
    X, y_true, y_weak = _weak_data()
    det = WSADDT(n_ensemble=2, seed=100).fit(X, y_weak)
    assert det.decision_scores_.shape == (len(X),)
    auc = roc_auc_score(y_true, det.decision_scores_)
    assert auc > 0.85, f"AUC too low: {auc}"
    assert set(np.unique(det.predict(X[:20]))) <= {0, 1}
    assert np.allclose(det.score_samples(X[:10]),
                       -det.decision_function(X[:10]))


def test_deterministic():
    X, _, y_weak = _weak_data(seed=1)
    a = WSADDT(n_ensemble=2, seed=100).fit(X, y_weak)
    b = WSADDT(n_ensemble=2, seed=100).fit(X, y_weak)
    assert np.allclose(a.decision_scores_, b.decision_scores_)


def test_equivalent_to_legacy_reference():
    """Wrapper reproduces the original WSAD_DT.py per-split training exactly."""
    from _legacy_original import (train_and_append_model as legacy_worker,
                                  evaluate_ensemble as legacy_eval,
                                  set_seed as legacy_set_seed)
    X, _, y_weak = _weak_data(seed=2)
    s, num_splits, batch_size = 100, 2, 64

    # --- legacy path (original code, serial over the same seed schedule) ---
    legacy_set_seed(s)
    c0 = np.where(y_weak == 0)[0]; c1 = np.where(y_weak == 1)[0]
    X0, y0, X1, y1 = X[c0], y_weak[c0], X[c1], y_weak[c1]
    subset = len(X0) // num_splits
    models, centers = [], []
    for i, seed in enumerate([s * (j + 1) for j in range(num_splits)]):
        m, c = legacy_worker(i, seed, subset, num_splits, X0, y0, X1, y1,
                             y_weak, X, batch_size, X.shape[1], 128, 'cpu')
        models.append(m); centers.append(c)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.zeros(len(X), dtype=torch.long)),
        batch_size=batch_size, shuffle=False)
    legacy_scores = legacy_eval(models, centers, loader, 'cpu')

    # --- packaged path ---
    det = WSADDT(n_ensemble=num_splits, seed=s).fit(X, y_weak)
    new_scores = det.decision_function(X)

    assert np.allclose(legacy_scores, new_scores, atol=1e-6), \
        "packaged WSADDT diverges from the original reference code"
