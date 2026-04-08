from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    ui_file = Path(__file__).resolve().parent / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(ui_file)]
    raise SystemExit(subprocess.call(command))
