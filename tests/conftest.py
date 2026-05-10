from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env
