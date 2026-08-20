"""WSAD-DT: Weakly Supervised Anomaly Detection via Dual-Tailed Kernel (ICML 2025)."""
from ._model import WSADDT
from ._reference import test, train

__version__ = "1.0.0"
__all__ = ["WSADDT", "train", "test", "__version__"]
