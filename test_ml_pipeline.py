import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "quality_gate.py"

def test_quality_gate_passes(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"accuracy": 0.95})
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

def test_quality_gate_fails(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"accuracy": 0.60})
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )

    assert result.returncode != 0
