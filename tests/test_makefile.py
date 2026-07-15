"""Tests for developer command interpreter selection."""

from __future__ import annotations

import subprocess


def test_make_test_uses_project_virtualenv() -> None:
    result = subprocess.run(
        ["make", "-n", "test"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ".venv/bin/python -m pytest"
