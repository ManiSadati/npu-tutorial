"""Local entry point for the shared baseline compare implementation."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "common" / "compare.py"), run_name="__main__")
