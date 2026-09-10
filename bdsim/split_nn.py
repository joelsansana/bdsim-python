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
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import torch
from torch import nn


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


class DecanterSplitNet(nn.Module):
    """Trainable PyTorch port of the decanter split MLP.

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
        super().__init__()
        wh = weights_hidden if weights_hidden is not None else _PESOS_W_HIDDEN
        wo = weights_out if weights_out is not None else _PESOS_W_OUT
        bh = bias_hidden if bias_hidden is not None else _BIAS_THETA
        bo = bias_out if bias_out is not None else _BIAS_GAMMA

        # Match the MATLAB forward pass exactly: linear → tanh → linear
        self.fc1 = nn.Linear(3, 5, bias=True)
        self.fc1.weight = nn.Parameter(torch.tensor(wh, dtype=torch.float32))
        self.fc1.bias = nn.Parameter(torch.tensor(bh, dtype=torch.float32))
        self.fc2 = nn.Linear(5, 3, bias=True)
        self.fc2.weight = nn.Parameter(torch.tensor(wo, dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor(bo, dtype=torch.float32))

        # Buffer for input normalisation (matches the MATLAB mn, st constants)
        self.register_buffer("mn", torch.tensor(_MN, dtype=torch.float32))
        self.register_buffer("st", torch.tensor(_ST, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., 3)`` tensor with columns ``[xM, xG, T_K]``.

        Returns
        -------
        torch.Tensor
            ``(..., 3)`` split fractions in the order ``[eta_E, eta_M, eta_G]``.
        """
        x_norm = (x - self.mn) / self.st
        h = torch.tanh(self.fc1(x_norm))
        return self.fc2(h)

    def freeze_pretrained(self) -> None:
        """Disable gradient on the upstream weights (useful for fine-tuning)."""
        for p in self.parameters():
            p.requires_grad_(False)


def split(xM: float, xG: float, T_K: float) -> np.ndarray:
    """Numpy reference evaluation of the upstream split MLP.

    Returns ``[eta_E, eta_M, eta_G]`` in the same order as the MATLAB
    ``split.m``. Useful for unit tests that don't need the full PyTorch
    overhead.
    """
    X = np.array([xM, xG, T_K])
    X = (X - _MN) / _ST
    hidden = np.tanh(_PESOS_W_HIDDEN @ X + _BIAS_THETA)
    return _PESOS_W @ hidden + _BIAS_GAMMA