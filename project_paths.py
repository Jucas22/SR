from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

BACKEND_DIR = ROOT_DIR / "Backend"
FRONTEND_DIR = ROOT_DIR / "Frontend"
DATA_DIR = ROOT_DIR / "Data"

RAW_DATA_DIR = DATA_DIR / "Raw_data"
CLEAN_DATA_DIR = DATA_DIR / "Clean_data"
POSTERS_DIR = DATA_DIR / "posters"
USER_REGISTRY_PATH = DATA_DIR / "user_registry.json"

COLLABORATIVE_DIR = BACKEND_DIR / "Colaborativo"
COLLABORATIVE_MATRIX_PATH = COLLABORATIVE_DIR / "preference_matrix.npz"
