import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if __name__ == "__main__":
    runpy.run_module("scripts.diagnostics.content_recommender_smoke", run_name="__main__")
