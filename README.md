# WSAD-DT — Weakly Supervised Anomaly Detection via Dual-Tailed Kernel

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://proceedings.mlr.press/v267/durani25a.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/Walid10010/Weakly-Supervised-Anomaly-Detection-via-Dual-Tailed-Kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/Walid10010/Weakly-Supervised-Anomaly-Detection-via-Dual-Tailed-Kernel/actions)

Official implementation of the ICML 2025 paper
**"Weakly Supervised Anomaly Detection via Dual-Tailed Kernel"**
(Walid Durani, Tobias Nitzl, Claudia Plant, Christian Böhm).

WSAD-DT learns latent representations that separate anomalies from normal
samples using **only a handful of labeled anomalies**. It introduces two
centroids — one normal, one anomalous — and a **dual-tailed kernel scheme**:
a light-tailed kernel compactly models in-class points while a heavy-tailed
kernel maintains a wide margin against out-of-class instances; a
kernel-based regularizer preserves intra-class diversity. An **ensemble**
partitions the normal data across members (all members share the labeled
anomalies), improving robustness.

## Install

```bash
pip install wsad-dt
```

or from source:

```bash
git clone https://github.com/Walid10010/Weakly-Supervised-Anomaly-Detection-via-Dual-Tailed-Kernel.git
cd Weakly-Supervised-Anomaly-Detection-via-Dual-Tailed-Kernel
pip install -e .
```

Runtime dependencies: `numpy`, `scikit-learn`, `torch` (CPU is sufficient).

## Quickstart

```python
from wsad_dt import WSADDT

# y_weak: 1 for the few labeled anomalies, 0 for everything else
det = WSADDT(n_ensemble=5, seed=100).fit(X_train, y_weak)

scores = det.decision_function(X_test)   # higher = more anomalous
labels = det.predict(X_test)             # 1 = anomaly
```

Inputs should be scaled (the paper protocol uses MinMax to [0, 1]).
Key parameters: `n_ensemble=5` (paper's `num_splits`), `seed=100`
(member *j* trains with seed `seed * (j+1)`), `batch_size=64`. The paper's
`train(num_splits, X, y_semi, s)` / `test(...)` entry points remain
importable from `wsad_dt` for script-level use.

## Reproducing the paper

Experiments use the [ADBench](https://github.com/Minqi824/ADBench) datasets
and compare against DeepSAD, DevNet, RoSAS, PReNet, GANomaly, and XGBOD:

```bash
pip install -r requirements-experiments.txt
python experiments/run_exp.py
```

Protocol: MinMax scaling, stratified 70/30 splits (`random_state` ∈ {1, 5, 10}),
**5% of anomalies labeled (minimum 5)**, ensemble of 5, AUC-ROC and AUC-PR
averaged over seeds and splits.

## Implementation notes

The packaged model is verified **equivalent to the original reference
script** (`tests/test_wsaddt.py::test_equivalent_to_legacy_reference`),
with training running deterministically on CPU. Scoring convention follows
PyOD: higher `decision_function` values indicate anomalies
(`score_samples` gives the scikit-learn sign convention). See
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for documented reference
semantics.

## Citation

```bibtex
@inproceedings{durani2025wsaddt,
  title     = {Weakly Supervised Anomaly Detection via Dual-Tailed Kernel},
  author    = {Durani, Walid and Nitzl, Tobias and Plant, Claudia and B{\"o}hm, Christian},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {14833--14866},
  publisher = {PMLR},
  year      = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
