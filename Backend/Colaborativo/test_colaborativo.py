import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.diagnostics.collaborative_recommender_smoke import (
    build_recommender,
    main,
    refresh_user_matrix,
    run_smoke_test,
)

__all__ = [
    "build_recommender",
    "refresh_user_matrix",
    "run_smoke_test",
    "main",
]


if __name__ == "__main__":
    main()
