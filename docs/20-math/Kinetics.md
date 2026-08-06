---
tags: [math]
aliases: [rxrates, Transesterification]
---

# Kinetics

Module: `bdsim/kinetics.py` (`rxrates`). JIT mirror inside [[ODE-and-AE]].

## Reaction network

Three reversible steps (TG ↔ DG ↔ MG → glycerol + esters):

$$
\begin{aligned}
\mathrm{TG} + \mathrm{M} &\rightleftharpoons \mathrm{DG} + \mathrm{E} \\
\mathrm{DG} + \mathrm{M} &\rightleftharpoons \mathrm{MG} + \mathrm{E} \\
\mathrm{MG} + \mathrm{M} &\rightleftharpoons \mathrm{G} + \mathrm{E}
\end{aligned}
$$

## Arrhenius rates

For each elementary direction \(i = 0..5\):

$$
k_i = k_{0,i}\,\exp\!\left(-\frac{E_{a,i}}{R\,T}\right)
$$

Net step rates \(r_0, r_1, r_2\) and species rates \(r_x\) (TG…G) match the arrays in `rxrates` / `_rxrates_jit`.

## Pseudocode

```text
kr[i] = k0[i] * exp(-Ea[i] / (R * T))
r0 = kr[0]*C_TG*C_M - kr[1]*C_DG*C_E
r1 = kr[2]*C_DG*C_M - kr[3]*C_MG*C_E
r2 = kr[4]*C_MG*C_M - kr[5]*C_G*C_E
rx_TG = -r0
rx_DG =  r0 - r1
...
rx_E  =  r0 + r1 + r2
```

Side-reaction deactivation scales pre-exponentials via [[Thermo|`side_reactions`]] (`ratio_robs_r` in `ProcessFaults`).

Related: [[Units-and-streams]], [[ODE-and-AE]], [[Thermo]], [[Acronyms]]
