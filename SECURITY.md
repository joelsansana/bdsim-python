# Security Policy

## Supported Versions

bdsim is currently maintained as a single release line — **the latest tagged
release** receives security updates. Older versions are not patched.

| Version | Supported          |
|---------|--------------------|
| 1.2.x   | :white_check_mark: |
| < 1.2   | :x:                |

## Reporting a Vulnerability

**Please do not file public issues for security problems.**

Email **joelsansana@gmail.com** with:

- A short description of the vulnerability
- Steps to reproduce (a minimal example, ideally a `run_with(...)` or
  `LiveSimulator(...)` snippet)
- Expected vs actual behaviour
- Any relevant stack traces

You should receive an acknowledgement within 72 hours. We will follow up with a
fix timeline once we have assessed the report.

## Disclosure Policy

- We follow a **coordinated disclosure** model.
- Once a fix is shipped, the reporter is credited (unless they prefer to remain
  anonymous).
- Critical issues will be backported where reasonable; non-critical fixes land
  in the next release.

## Scope

The package itself is **pure-Python research code** that loads reference spectra
from a bundled CSV (`bdsim/data/spectra_ref.csv`) and writes plot HTML to a
results directory. It does not:

- Listen on any network port
- Open files outside the user-specified `--outdir` and `spectra_ref_path`
- Spawn subprocesses
- Make outbound HTTP requests

If you find a path that does any of those things unintentionally, treat it as a
high-severity bug and report privately.

The companion [bdsim-dashboard](https://github.com/joelsansana/bdsim-dashboard)
repo has its own security policy.
