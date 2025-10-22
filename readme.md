
# Weakly Supervised Anomaly Detection via Dual-Tailed Kernel (WSAD-DT)

WSAD-DT is a **weakly supervised anomaly detection** framework that introduces a **dual-tailed kernel mechanism** to achieve robust separation between normal and anomalous samples, even when only a small fraction of anomalies are labeled.
This repository supports benchmarking against multiple baseline models (DeepSAD, DevNet, RoSAS, PReNet, GANomaly, and XGBOD) across standard tabular datasets.

# Paperpage:
https://icml.cc/virtual/2025/poster/44385
---

## 📦 Repository Structure

```
├── WSAD_DT.py          # WSAD-DT model: training and testing functions
├── run_exp.py          # Experiment runner for all algorithms and datasets
├── requirements.txt    # Python dependencies
```

---

##  Installation

Ensure Python **= 3.9** and install dependencies via:

```bash
pip install -r requirements.txt
```

If running all baselines, also install optional dependencies:

```bash
pip install torch pyod pandas tqdm
```

---

## 🚀 Running Experiments

The main entry point is **`run_exp.py`**, which automatically iterates through available datasets and algorithms.

### Run all algorithms

```bash
python run_exp.py
```

##  Dataset

Experiments are based on the **[ADBench benchmark](https://github.com/Minqi824/ADBench/tree/main)**.

Download and prepare data:

```bash
git clone https://github.com/Minqi824/ADBench.git
```



## 🧩 Parameters and Settings

Although the script does not use CLI arguments, internal parameters can be adjusted directly inside `run_exp.py`:

| Parameter             | Description                                    | Default       |
| --------------------- | ---------------------------------------------- | ------------- |
| `label_percentage`    | Ratio of labeled anomalies                     | `0.05`        |
| `mislabel_percentage` | Ratio of mislabeled normal samples             | `0.0`         |
| `contamination_ratio` | Ratio of unlabeled anomalies added to training | `0.0`         |
| `batch_size`          | Training batch size                            | `64`          |
| `seeds`               | Random seeds used                              | `[100, 1000]` |
| `ss`                  | Train/test random split seeds                  | `[1, 5, 10]`  |

**Note:** You can modify these directly in the script for experiments (e.g., increase labeled anomalies or seeds).

---

## 🧮 Metrics

Two evaluation metrics are reported for each dataset:

* **AUC-ROC** — Area under the Receiver Operating Characteristic curve
* **Average Precision (AUC-PR)** — Area under the Precision-Recall curve

Both are averaged over multiple seeds and random splits.

---

## 📊 Output

After execution, you’ll see output like:

```
Dataset: Classical/mnist.npz
WSAD_DT - Average AUC-ROC: 0.9874
WSAD_DT - Average Precision Score: 0.9821
```

To save results automatically, uncomment lines 166–170 in `run_exp.py`:
This will store results under `icml_results/`.

---


## 🧠 How WSAD-DT Works

1. **Dual-tailed kernel loss** — combines a light-tailed Gaussian and a heavy-tailed Student-t kernel

   * Compact clusters for normal data
   * Robust separation for anomalies
2. **Center learning** — normal and anomaly centers are dynamically updated
3. **Diversity regularization** — prevents latent collapse
4. **Ensemble strategy** — unlabeled data are split into subsets while labeled anomalies are shared



---
