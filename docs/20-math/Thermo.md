---
tags: [math]
aliases: [Qoil, Mixture properties]
---

# Thermo

Module: `bdsim/thermo.py` — pure NumPy leaf functions (easy to unit-test). JIT copies of critical bits live in [[ODE-and-AE]].

## Functions

| Function | Role | Units |
|----------|------|-------|
| `Qoil(r, α, p)` | Oil volumetric flow through filter | m³/s |
| `Vmolar(M, ρ)` | Molar volume | m³/mol |
| `Mmx(M, x)` | Mixture molar mass | kg/mol |
| `cpmx(cp, x)` | Mixture heat capacity | J/(mol·K) or massic |
| `side_reactions(A, ratio)` | Scale pre-exponential factors | dimensionless |

## Filter flow (sketch)

Pore radius \(r\) and valve opening fraction \(\alpha\):

$$
Q_{\mathrm{oil}} = -K_{2F}\frac{\alpha^{2}}{r^{4}} + \sqrt{K_{2F}^{2}\frac{\alpha^{4}}{r^{8}} + K_{3F}\,\alpha^{2}}
$$

Constants `K2F`, `K3F`, `K4F` come from `Parameters` (filter geometry). Mass flow measurement uses \(Q_{\mathrm{oil}}\cdot\rho_{\mathrm{oil}}\) — [[Sensors-and-control]].

## Mixture rules

Mole-fraction weighted means for \(M\) and \(c_p\) — callers must keep molar vs massic consistent.

Related: [[Kinetics]], [[ODE-and-AE]], [[Process-flow]], [[Units-and-streams]]
