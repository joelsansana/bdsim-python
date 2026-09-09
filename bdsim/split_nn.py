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
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Hardcoded weights and biases from split.m (numerically faithful port)
_MN = np.array([0.4429080932784634, 0.1430123456790114, 315.4873799725652])
_ST = np.array([0.05769114669974407, 0.07483374254327435, 10.3923004820672])

_PESOS_W = np.array([
    [ 0.0001626573488477621,  0.0001762383521689282,  0.008436427201807458,  0.0001270214941574136, 0.004297765775023732],
    [ 0.005798439102391745,   0.007933451534695593,  2.26649586627043,    -0.2986375856202046,    0.7311754343879735],
    [-0.001364318678976015,   0.0007368268046611492, 0.3912844089747987,   0.002646628766137311,   0.1563496596243398],
])
_PESOS_W_OUT = _PESOS_W.copy()                                # PyTorch nn.Linear weight is (out, in)

_PESOS_W_HIDDEN = np.array([
    [-0.2803986419064017,  -0.1333039855614737, -0.4777087005027253],
    [-1.089354327582992,   -0.569687192601498,   0.2811677238454339],
    [-0.08556871701830951, -0.9941218009650442,  0.03066753440441023],
    [ 0.1392959274375281,   0.3224633500842102, -0.06143635056503905],
    [ 0.07424143386924453,  1.155133146934136,  -0.01735683560849739],
])
# _PESOS_W_HIDDEN is already (5, 3), matching nn.Linear(3, 5) weight shape

_BIAS_THETA = np.array([0.6227988072448851, 1.158840062875017, -2.463591740961228,
                       0.2729954333628518, 2.246274101076976])
_BIAS_GAMMA = np.array([1.003727297064683, 1.891158698301678, 0.2350237516771797])


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