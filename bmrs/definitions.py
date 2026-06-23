import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = ROOT_DIR.parent / "configs"
RESULTS_DIR = ROOT_DIR.parent / "results"
