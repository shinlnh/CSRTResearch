"""
Run CSRT Update (Hybrid CSRT-Deep) Training
With virtual environment activation
"""

import subprocess
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).parent
venv_python = project_root / "venv" / "Scripts" / "python.exe"

if not venv_python.exists():
    print(f"Error: Virtual environment not found at {venv_python}")
    print("Please create venv first: python -m venv venv")
    sys.exit(1)

# Arguments for training
args = [
    str(venv_python),
    str(project_root / "update_csrt" / "train.py"),
    "--dataset-root", str(project_root / "otb100"),
    "--batch-size", "4",
    "--num-epochs", "5",
    "--num-workers", "0",
    "--device", "cpu",
    "--save-dir", str(project_root / "update_csrt" / "checkpoints"),
    "--log-dir", str(project_root / "update_csrt" / "runs")
]

print("="*80)
print("Starting CSRT Update Training")
print("="*80)
print(f"Python: {venv_python}")
print(f"Script: update_csrt/train.py")
print(f"Dataset: otb100/")
print("="*80)

# Run training
subprocess.run(args)
