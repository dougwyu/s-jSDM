import subprocess
import sys
from pathlib import Path


def test_byte_gate_survives_optimized_python():
    validator = Path(__file__).with_name("validate_mojo_release.py")
    code = f"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "validate_mojo_release", {str(validator)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def explicit_result(path, shape, seed):
    value = 0.0 if path == "baseline" else 1.0
    return (np.array([value], dtype=np.float32),)

module.explicit_result = explicit_result
module.check_cross_binary_bytes("baseline", "candidate")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, result.stdout
    assert "response bytes changed" in result.stderr
