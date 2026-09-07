"""Shared pytest configuration for the bdsim test suite.

Fingerprint pins in this repo are calibrated to the documented reference
stack (Python 3.10, numpy 2.2.6, scipy 1.15.3, numba 0.66.0 — see
``uv.lock`` and ``ADMIN.md``). On Python 3.11+ the lockfile resolves to
a different numpy/scipy, and the trajectory hash diverges even when the
math is unchanged.

The ``fingerprint_reference_stack`` marker skips a test on any
interpreter other than the reference. Apply it to fingerprint-pinned
regression tests; do not apply it to functional / unit tests, which
should pass on every supported interpreter.
"""

from __future__ import annotations

import sys

import pytest

REFERENCE_PYTHON = (3, 10)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fingerprint_reference_stack: skip on Python != 3.10 (uv.lock reference stack)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if sys.version_info[:2] == REFERENCE_PYTHON:
        return
    skip_marker = pytest.mark.skip(
        reason=f"fingerprint pinned to the reference stack (Python {REFERENCE_PYTHON[0]}.{REFERENCE_PYTHON[1]}); see conftest.py"
    )
    for item in items:
        if "fingerprint_reference_stack" in item.keywords:
            item.add_marker(skip_marker)
