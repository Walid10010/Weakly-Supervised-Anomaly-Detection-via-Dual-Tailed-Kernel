"""WSADDT: scikit-learn-style wrapper around the WSAD-DT reference code.

Reference:
    W. Durani, T. Nitzl, C. Plant, C. Boehm.
    "Weakly Supervised Anomaly Detection via Dual-Tailed Kernel." ICML 2025.
    https://proceedings.mlr.press/v267/durani25a.html

Score convention (PyOD-style): higher score = more anomalous.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from ._reference import evaluate_ensemble, set_seed, train

__all__ = ["WSADDT"]


class WSADDT(BaseEstimator):
    """Weakly Supervised Anomaly Detection via Dual-Tailed Kernel.

    Trains an ensemble of encoders with a dual-tailed kernel loss: a
    light-tailed kernel models in-class compactness around a normal and an
    anomalous centroid, while a heavy-tailed kernel keeps a wide margin
    against out-of-class points; a kernel-based regularizer preserves
    intra-class diversity. Only a small set of labeled anomalies is needed.

    Parameters
    ----------
    n_ensemble : int, default=5
        Number of ensemble members (the paper's ``num_splits``); normal
        training data is partitioned across members, labeled anomalies are
        shared by all.
    seed : int, default=100
        Base random seed ``s``; member ``j`` uses seed ``s * (j + 1)``,
        as in the paper script.
    contamination : float, default=0.1
        Expected anomaly proportion; sets ``threshold_``/``labels_`` only.
    batch_size : int, default=64
        Mini-batch size for training and scoring (paper default).
    parallel : bool, default=False
        If True, train members in a multiprocessing pool exactly like the
        paper script; if False (default), train serially — numerically
        identical, since every member reseeds all RNGs.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Anomaly scores of the training data (higher = more anomalous).
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
        Binary training labels implied by ``contamination`` (1 = anomaly).

    Notes
    -----
    Training runs on CPU (as in the reference implementation) with
    deterministic algorithms enabled. Inputs are expected to be scaled;
    the paper protocol uses MinMax scaling to [0, 1].

    Examples
    --------
    >>> det = WSADDT(n_ensemble=5, seed=100).fit(X_train, y_weak)
    >>> scores = det.decision_function(X_test)   # higher = more anomalous
    """

    def __init__(self, n_ensemble=5, seed=100, contamination=0.1,
                 batch_size=64, parallel=False):
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.contamination = contamination
        self.batch_size = batch_size
        self.parallel = parallel

    def fit(self, X, y):
        """Fit on weakly labeled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (scale beforehand; MinMax [0, 1] recommended).
        y : array-like of shape (n_samples,)
            Weak labels: 1 for the few *labeled anomalies*, 0 for all other
            (unlabeled, presumed-normal) points.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        y = y.astype(int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must contain only 0 (unlabeled/normal) and "
                             "1 (labeled anomaly).")
        if (y == 1).sum() < 1 or (y == 0).sum() < self.n_ensemble:
            raise ValueError(
                "Need at least 1 labeled anomaly and at least n_ensemble "
                "unlabeled/normal samples.")

        X = np.asarray(X, dtype=np.float64)
        set_seed(self.seed)
        self.models_, self.centers_ = train(
            self.n_ensemble, X, y, self.seed, parallel=self.parallel)

        self.n_features_in_ = X.shape[1]
        self.decision_scores_ = self.decision_function(X)
        self.threshold_ = np.percentile(
            self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        """Anomaly scores for ``X`` (higher = more anomalous)."""
        check_is_fitted(self, "models_")
        X = check_array(X)
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor, torch.zeros(len(X_tensor),
                                                dtype=torch.long)),
            batch_size=self.batch_size, shuffle=False)
        return evaluate_ensemble(self.models_, self.centers_, loader, 'cpu')

    def score_samples(self, X):
        """Scores with scikit-learn sign convention (higher = more normal)."""
        return -self.decision_function(X)

    def predict(self, X):
        """Binary labels for ``X`` (1 = anomaly), using ``threshold_``."""
        check_is_fitted(self, "threshold_")
        return (self.decision_function(X) > self.threshold_).astype(int)

    def fit_predict(self, X, y):
        """Fit on weakly labeled ``(X, y)`` and return training labels."""
        return self.fit(X, y).labels_
