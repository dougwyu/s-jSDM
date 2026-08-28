"""Portability checks for the Mojo bridge test module itself."""

from pathlib import Path
import subprocess
import sys


def test_mojo_bridge_tests_import_without_resource_module():
    """Unsupported platforms must collect the binary-gated Mojo tests."""
    inst_dir = Path(__file__).resolve().parents[2]
    script = """
import sys

import numpy
import pytest
import torch

sys.modules["resource"] = None
from python.tests import test_mojo_bridge
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=inst_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
