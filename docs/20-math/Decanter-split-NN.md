---
tags: [math]
aliases: [split.m, DecanterSplitNet, Phase split]
---

# Decanter-split-NN

Module: `bdsim/split_nn.py`. Port of Brásio / Romanenko / Fernandes split MLP (MATLAB `split.m`).

## Role in the plant

Predicts split fractions \(\eta_E, \eta_M, \eta_G\) that close the decanter mass balances inside [ODE-and-AE](../20-math/ODE-and-AE.md). Evaluated **outside** the Numba JIT block once per step (weights are small; transition is amortized).

## Architecture

Inputs: \(X = [x_M, x_G, T]\) (normalized with stored mean/std).

```text
X_n = (X - mn) / st
h   = tanh(W_h @ X_n + θ)     # 5 hidden units
η   = W_o @ h + γ              # 3 outputs: eta_E, eta_M, eta_G
```

```mermaid
flowchart LR
  X[xM xG T] --> norm[normalize]
  norm --> hid[tanh hidden 5]
  hid --> out[eta_E eta_M eta_G]
```

## Two APIs

- `split(...)` — NumPy evaluation with **hardcoded** upstream weights (default sim path)
- `DecanterSplitNet` — PyTorch `nn.Module` for training / research

> [!note]
> Weights live in source, not a separate `.pt` file. Treat numerical values as part of the faithful port.

Related: [ODE-and-AE](../20-math/ODE-and-AE.md), [Process-flow](../10-plant/Process-flow.md), [Units-and-streams](../10-plant/Units-and-streams.md), [Acronyms](../Acronyms.md#mlp)
