---
tags: [helper]
aliases: [Doc pseudocode style]
---

# Pseudocode-conventions

How this vault writes algorithms so juniors can map notes → Python.

## Rules

1. Use plain fenced `text` blocks (not a fake language) unless showing real Python API
2. Name arrays like the code: `sv`, `pv`, `uv`, `sp`
3. Mark JIT boundaries: “outside JIT” vs “`_ode_rhs_jit`”
4. Prefer one step of the driver loop over a full file dump
5. Link the implementing note: e.g. [Batch-driver](../30-engine/Batch-driver.md), [ODE-and-AE](../20-math/ODE-and-AE.md)

## Example shape

```text
# Good: one concern
for each control period:
    e = sp - pv
    u = PID(e)
    sv = integrate(ODE, sv, u, dt)
    pv = measure(sv) + sensor_faults
```

## Equations

Use LaTeX `$$ ... $$` for balances and kinetics; keep symbols consistent with [Units-and-streams](../10-plant/Units-and-streams.md) and [Acronyms](../Acronyms.md).

## Diagrams

Mermaid `flowchart` for plant/module maps; `sequenceDiagram` for live step / PID. Keep node IDs camelCase without spaces.

Related: [Home](../Home.md), [Practical dev workflow](../../AGENTS.md), [Batch-driver](../30-engine/Batch-driver.md), [Live-simulator](../30-engine/Live-simulator.md)
