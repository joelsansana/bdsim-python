# Pull Request

Thanks for contributing! Please fill in the sections that apply.

## What

<!-- One-sentence summary of the change. -->

## Why

<!-- The motivation. What bug does it fix or feature does it add? -->

## Fingerprint impact

- [ ] No fingerprint change (default-off / no ODE change)
- [ ] Fingerprint bump — old pin ↔ new pin:

| profile  | old               | new               |
|----------|-------------------|-------------------|
| legacy   | `<hash>…`         | `<hash>…`         |
| dynamic  | `<hash>…`         | `<hash>…`         |
| runtime  | `<hash>…`         | `<hash>…`         |

## Tests

- [ ] `uv run python -m pytest tests/ -q` passes locally
- [ ] New test added (if behaviour changed)
- [ ] `uv run ruff check bdsim/ tests/` clean

## Docs

- [ ] Updated [`Config-surface.md`](docs/30-engine/Config-surface.md) (if a
      `ProcessFaults` field was added / renamed)
- [ ] Updated [`Channel-indices.md`](docs/40-helpers/Channel-indices.md) (if
      `sv` / `pv` / `uv` widths changed)
- [ ] Updated [`Byte-identical-contract.md`](docs/00-orientation/Byte-identical-contract.md)
      (if a new pinned profile was added)
- [ ] Updated [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`

## Notes for reviewers

<!-- Anything else reviewers should know: tradeoffs, follow-ups, etc. -->

---

By submitting this PR you agree your contributions are licensed under GPLv3+
(the same license as the project; see [`LICENSE`](LICENSE)).
