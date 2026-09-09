# Contributing to bdsim

Thanks for your interest in contributing! This project is a faithful Python port of
Natércia Fernandes' MATLAB/Octave BDSIM, with deterministic batch and live APIs
for fault-detection research and operator-training dashboards.

Before opening an issue or PR, please skim:

- **[README.md](README.md)** — install, run, programmatic use.
- **[USER.md](USER.md)** — recipes and quick-starts for common tasks.
- **[AGENTS.md](AGENTS.md)** — hard rules and the canonical dev workflow.
  Treat the "What NOT to do without asking Joel" list as a hard gate.
- **[ADMIN.md](ADMIN.md)** — install / fingerprint-aligned env setup.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** — community standards.
- **[CHANGELOG.md](CHANGELOG.md)** — release history and pinned-profile
  announcements.
- **[SECURITY.md](SECURITY.md)** — how to report vulnerabilities privately.

## How to add a knob

Most contributions add or tweak a `ProcessFaults` knob. The pattern:

1. **Pick a name.** Match the existing field-naming convention (`<feature>_<thing>`
   snake_case). Don't reuse a name from a different feature group without good
   reason.
2. **Default off / zero / `None`.** Trajectory-affecting behaviour must default
   to a value that reproduces the current fingerprint. The one documented
   exception is `fouling_dynamic=True`; check with maintainer before adding
   another.
3. **Wire it through the Numba kernel only if it must affect the ODE.** If it's
   a post-process knob (e.g. sensor noise, plot colour), keep it out of the
   `@njit(cache=True)` path — the fingerprints depend on the kernel order.
4. **Document it.** Update the `ProcessFaults` table in
   [`docs/30-engine/Config-surface.md`](docs/30-engine/Config-surface.md).
   If you widen the state vector, also update
   [`docs/40-helpers/Channel-indices.md`](docs/40-helpers/Channel-indices.md).
5. **Add a test.** Either add a fingerprint pin to an existing profile (only
   if you intentionally changed the trajectory) or a unit test for the new
   feature. Tests live in `tests/`.
6. **Run the suite.** `uv run python -m pytest tests/ -q` (~6:40).

## How to bump a fingerprint

If your change is intentional and affects the trajectory, the pinned SHA-256
hashes in `tests/` must move with it. The protocol:

1. Confirm the trajectory change is approved (ODE changes need explicit
   maintainer sign-off — see "What NOT to do" in `AGENTS.md`).
2. Re-run `uv run python -m pytest tests/ -q`; copy the new hashes the suite
   prints.
3. Update the pin in the relevant test file.
4. In the same commit, the commit body must contain a "this is intentional"
   line:
   ```
   fingerprint: legacy sv: 6f61eb53... -> c8807b23...
   fingerprint: live  sv: f37fb5e0... -> 23c3c885...
   ```
   This is non-negotiable. Reviewers and `git log` readers use it to spot
   drift at a glance.
5. Bump the version in `pyproject.toml` **and** `bdsim/__init__.py` (they are
   pinned together). A fingerprint bump is a minor version bump at minimum.

## Asking for a review

- Open a PR against `main`. Branch naming: `feature/<name>`, `fix/<name>`,
  `chore/<name>`.
- Commit messages: `feat(<scope>)`, `fix(<scope>)`, `chore(<scope>)` prefix.
- The PR description should call out: what changed, why, and any fingerprint
  updates (old → new pins).
- Tests + `uv run ruff check bdsim/ tests/` must be clean.

## Reporting bugs

Use the [GitHub issue tracker](https://github.com/joelsansana/bdsim-python/issues).
For security issues, see [SECURITY.md](SECURITY.md) — please do not file
publicly.

## License

By contributing, you agree that your contributions will be licensed under
GPLv3+ (the same license as the project; see [LICENSE](LICENSE)).
