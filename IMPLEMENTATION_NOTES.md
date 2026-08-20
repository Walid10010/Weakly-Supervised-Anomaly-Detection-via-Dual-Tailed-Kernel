# Implementation notes — reference semantics of WSAD-DT

The packaged implementation (`wsad_dt/_reference.py` + `wsad_dt/_model.py`)
is **behavior-preserving** with respect to the original `WSAD_DT.py` used
for the ICML 2025 experiments: for identical data and seeds it produces
equivalent ensembles and scores (see
`tests/test_wsaddt.py::test_equivalent_to_legacy_reference`). Non-behavioral
changes only: imports consolidated, module-level side effects
(`torch.use_deterministic_algorithms(True)`, `mp.set_start_method('spawn')`)
moved into `set_seed()` / `train()`, dead code paths removed.

Aspects of the reference semantics worth knowing before extending or
porting the method:

1. **Kernel/tail naming vs. behavior.** The methods named `light_trail` /
   `lightt_tail_n` return Gaussian-type kernels (`exp(-d²)` and
   `exp(-d²/0.5)`), while `heavy_trail` returns the Student-t kernel with
   α = 0.2. In `compute_similarity`, in-class points therefore use the
   *Gaussian* (light-tailed) kernels — `exp(-d²/0.5)` toward the normal
   center, `exp(-d²)` toward the anomalous center — while out-of-class
   points use the *t-distribution* (heavy-tailed) kernel, matching the
   paper's light-in / heavy-out design even though several docstrings in
   the original file describe the functions inversely.
2. **`alpha` constructor argument is inert.** `TDistributionLoss(...,
   alpha=100)` overrides alpha to 0.2 in `__init__` and again inside each
   tail function; evaluation (`evaluate_ensemble`) also hard-codes 0.2.
3. **Fixed hyperparameters in the reference path.** Encoder 100→50→128
   (bias-free linear layers, SELU), latent_dim=128, Adam lr=1e-3,
   weight_decay=1e-5, grad-clip 1.0, 100 epochs, batch_size=64, and
   `device='cpu'` are hard-coded in `f()`/`train()`.
4. **Sampling asymmetry.** The `WeightedRandomSampler` uses weights
   `{0: 2/count_0, 1: 1/count_1}` — unlabeled/normal points receive twice
   the per-class mass of labeled anomalies — with
   `num_samples = len(X_train)` and replacement.
5. **RNG-coupled regularizer.** `mm()` subsamples up to 8 points per class
   via `torch.randperm` each step; it consumes the global torch RNG and is
   part of the deterministic sequence (reproducibility requires the same
   seed schedule `s * (j + 1)`).
6. **NaN handling.** NaN batch losses are skipped; NaN class losses are
   zeroed; ensemble scoring uses `np.nanmean` across members.
7. **Center epsilon-push.** Initial centroids have coordinates with
   |value| < 0.1 pushed to ±0.1 before becoming trainable parameters.
8. **Unused arguments.** `TDistributionLoss.forward` receives
   `unique_latent_vectors` / `unique_labels` that are computed in the
   training loop but not used by the loss.
9. **Serial ≡ parallel.** `train(..., parallel=True)` reproduces the paper
   script's spawn-pool; the serial path is numerically identical because
   every member calls `set_seed(seed)` before any RNG use. The packaged
   `WSADDT` defaults to serial for portability.

## Recommended follow-ups

- If the tail functions are ever renamed to match their formulas
  (item 1), keep the current functions as aliases so published behavior
  and repository history stay reproducible.
- Any change to items 2–4 changes numbers; bump the package minor version
  and note it in a changelog rather than silently altering defaults.
