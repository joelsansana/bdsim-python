---
tags: [math]
aliases: [ODEmodel, State equations, AEmodel]
---

# ODE-and-AE

Plant dynamics live in `bdsim/ode.py`: Numba-compiled **RHS** (`ODEmodel`) plus algebraic helpers (`AEmodel`). Do not reorder `@njit` statements — see [Byte-identical-contract](../00-orientation/Byte-identical-contract.md).

## Baseline state vector (21)

Documented at top of `ode.py`:

| Slice | Symbol | Meaning |
|-------|--------|---------|
| `0:6` | xR | Reactor mole fractions (6 species) |
| `6` | TR | Reactor temperature, K |
| `7:13` | xL | Light-phase composition |
| `13:16` | xH | Heavy-phase M/E/G |
| `16` | hH | Heavy interface level, m |
| `17` | TD | Decanter temperature, K |
| `18` | r | Filter pore radius, m |
| `19` | lifto | Oil valve lift, % |
| `20` | liftH | Heavy valve lift, % |

Optional feature-flagged slots grow beyond 21 — [Channel-indices](../40-helpers/Channel-indices.md).

## Inputs `u` (6)

`vinputo`, `Tmet`, `Fmet`, `Toil`, `Qheat`, `vinputH` — see [Channel-indices](../40-helpers/Channel-indices.md).

## Pseudocode (one integration step)

```text
1. Evaluate decanter split NN → eta_E, eta_M, eta_G   # outside JIT
2. Pack parameters into JIT-friendly args
3. ds/dt = _ode_rhs_jit(t, sv, u, factor, …)
   - filter flow Qoil(r, valve)
   - reaction rates at TR ([Kinetics](../20-math/Kinetics.md))
   - mass / energy balances (reactor, decanter, valves)
   - optional: dα/dt, quality states, wear states
4. solve_ivp / step advances sv by dt
5. AE / washer-dryer post-process as in drivers
```

## HEX energy sketch

Fouling enters as an efficiency **factor** on heat duty (see `fouling_modes` module docstring):

$$
T_{\mathrm{heat}} \sim T_R - \mathrm{factor}\cdot\frac{Q_{\mathrm{heat}}}{N_R\, c_{p,\mathrm{mol},R}}
$$

`factor` comes from continuous α, windowed ARMAX modes, or static legacy series ([Config-surface](../30-engine/Config-surface.md)).

## AE model

Algebraic relations (not integrated states) — e.g. derived flows / washer-dryer outputs consumed by drivers when building `xLend` / `yLend`.

Related: [Kinetics](../20-math/Kinetics.md), [Thermo](../20-math/Thermo.md), [Decanter-split-NN](../20-math/Decanter-split-NN.md), [Repo-map](../00-orientation/Repo-map.md), [Practical dev workflow](../../AGENTS.md)
