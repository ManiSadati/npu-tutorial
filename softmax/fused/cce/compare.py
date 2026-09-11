"""Local entry point for the shared baseline compare implementation."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2] / "baseline" / "common" / "compare.py"), run_name="__main__")
