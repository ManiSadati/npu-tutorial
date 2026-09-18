"""Local entry point for the shared baseline generate implementation."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "common" / "generate.py"), run_name="__main__")
