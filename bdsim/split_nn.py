"""
Decanter split-fraction neural network.

Original model by Brásio, Romanenko, and Fernandes (Springer, 2014) —
DOI:10.1007/978-3-319-20328-7_6. Ported here as a small PyTorch MLP so that
it can be both evaluated numerically (with the upstream weights) and
re-trained against new data without touching the simulator source.

Architecture (matches the MATLAB ``split.m`` exactly):
    X = [xM, xG, T] in K
    X_normalized = (X - mn) / st
    hidden = tanh( pesos_w @ X + bias_theta )       # 5 hidden units
    output = pesos_W @ hidden + bias_gamma            # 3 outputs (eta for E, M, G)

Weights are loaded from the bundled ``bdsim/data/split_nn_weights.npz``
file (issue #12) so a retrain doesn't require editing Python source
— drop the new ``.npz`` next to the existing one and the simulator
picks it up at import time.

Optional dependency: ``torch`` is only required for the trainable
:class:`DecanterSplitNet` class. The pure-NumPy :func:`split` function
that the ODE kernel uses is **torch-free** — inference-only users
do not need to install the (hundreds-of-MB) ``torch`` package. See
issue #13.
"""

from __future__ import annotations

from importlib import resources

import numpy as np


def _load_upstream_weights() -> dict[str, np.ndarray]:
    """Load the upstream split-net weights from the bundled .npz.

    The file is loaded from the package data directory
    (``bdsim/data/split_nn_weights.npz``) and the same access pattern
    used for ``spectra_ref.csv`` (see :mod:`bdsim.spectra`).
    """
    from io import BytesIO
    blob = resources.files("bdsim.data").joinpath("split_nn_weights.npz").read_bytes()
    with np.load(BytesIO(blob)) as data:
        return {k: np.array(data[k], dtype=np.float64) for k in data.files}


_UPSTREAM = _load_upstream_weights()
_MN = _UPSTREAM["mn"]
_ST = _UPSTREAM["st"]
_PESOS_W = _UPSTREAM["pesos_w"]
_PESOS_W_OUT = _PESOS_W.copy()                                # PyTorch nn.Linear weight is (out, in)
_PESOS_W_HIDDEN = _UPSTREAM["pesos_w_hidden"]
_BIAS_THETA = _UPSTREAM["bias_theta"]
_BIAS_GAMMA = _UPSTREAM["bias_gamma"]


def _require_torch():
    """Return ``(torch, nn)`` or raise a friendly ImportError.

    Used as a guard inside :class:`DecanterSplitNet` so that
    ``import bdsim`` succeeds on a torch-less install and the
    :func:`split` NumPy path is still available. Training and
    fine-tuning workflows that need the trainable module must
    install the ``[torch]`` extra (``pip install bdsim[torch]``).
    """
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise ImportError(
            "DecanterSplitNet requires torch. Install with: "
            "pip install bdsim[torch]"
        ) from e
    return torch, nn


class DecanterSplitNet:
    """Trainable PyTorch port of the decanter split MLP.

    The class is **only defined when ``torch`` is importable** —
    see :data:`__all__` and :func:`_require_torch`. A pure-NumPy
    :func:`split` is the inference path the simulator actually
    uses; the PyTorch module exists for retraining and fine-tuning.

    Parameters
    ----------
    weights_hidden : np.ndarray, optional
        ``(3, 5)`` matrix to override the upstream ``pesos_w``. If provided,
        the model is initialised from this matrix rather than the upstream
        weights.
    weights_out : np.ndarray, optional
        ``(5, 3)`` matrix to override the upstream ``pesos_W``.
    bias_hidden : np.ndarray, optional
        ``(5,)`` bias vector for the hidden layer.
    bias_out : np.ndarray, optional
        ``(3,)`` bias vector for the output layer.

    The model is intentionally *not* frozen — call :meth:`freeze_pretrained`
    to disable gradient updates on the upstream weights when fine-tuning.
    """

    def __init__(
        self,
        weights_hidden: np.ndarray | None = None,
        weights_out: np.ndarray | None = None,
        bias_hidden: np.ndarray | None = None,
        bias_out: np.ndarray | None = None,
    ) -> None:
        torch, nn = _require_torch()
        self._nn_module = nn
        # The actual nn.Module instance is built in a private subclass
        # so the public class itself doesn't depend on torch at
        # class-definition time. We forward attribute access to it.
        self._impl = self._build_impl(
            torch, nn,
            weights_hidden, weights_out, bias_hidden, bias_out,
        )

    @staticmethod
    def _build_impl(torch, nn, wh, wo, bh, bo):
        """Construct the underlying ``nn.Module``."""

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                wh_use = wh if wh is not None else _PESOS_W_HIDDEN
                wo_use = wo if wo is not None else _PESOS_W_OUT
                bh_use = bh if bh is not None else _BIAS_THETA
                bo_use = bo if bo is not None else _BIAS_GAMMA
                # Match the MATLAB forward pass: linear → tanh → linear
                self.fc1 = nn.Linear(3, 5, bias=True)
                self.fc1.weight = nn.Parameter(torch.tensor(wh_use, dtype=torch.float32))
                self.fc1.bias = nn.Parameter(torch.tensor(bh_use, dtype=torch.float32))
                self.fc2 = nn.Linear(5, 3, bias=True)
                self.fc2.weight = nn.Parameter(torch.tensor(wo_use, dtype=torch.float32))
                self.fc2.bias = nn.Parameter(torch.tensor(bo_use, dtype=torch.float32))
                # Buffer for input normalisation (matches the MATLAB mn, st constants)
                self.register_buffer("mn", torch.tensor(_MN, dtype=torch.float32))
                self.register_buffer("st", torch.tensor(_ST, dtype=torch.float32))

            def forward(self, x):
                x_norm = (x - self.mn) / self.st
                h = torch.tanh(self.fc1(x_norm))
                return self.fc2(h)

        return _Impl()

    def __call__(self, x):
        """Make the wrapper callable like a normal ``nn.Module``."""
        return self._impl(x)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying nn.Module so the
        # public DecanterSplitNet looks like a thin wrapper.
        if name in {"_impl", "_nn_module"}:
            raise AttributeError(name)
        return getattr(self._impl, name)

    def freeze_pretrained(self) -> None:
        """Disable gradient on the upstream weights (useful for fine-tuning)."""
        for p in self._impl.parameters():
            p.requires_grad_(False)


def split(xM: float, xG: float, T_K: float) -> np.ndarray:
    """Numpy reference evaluation of the upstream split MLP.

    Returns ``[eta_E, eta_M, eta_G]`` in the same order as the MATLAB
    ``split.m``. Pure NumPy — does not require ``torch``.
    """
    X = np.array([xM, xG, T_K])
    X = (X - _MN) / _ST
    hidden = np.tanh(_PESOS_W_HIDDEN @ X + _BIAS_THETA)
    return _PESOS_W @ hidden + _BIAS_GAMMA
