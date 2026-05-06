from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
DATA_DIR = PROJECT_ROOT / "data"
CAMPUS_KB_DIR = DATA_DIR / "campus_kb"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
CAMPUS_INDEX_DIR = VECTOR_STORE_DIR / "campus_kb"


def ensure_project_dirs() -> None:
    for path in (
        ARTIFACTS_DIR,
        PREDICTIONS_DIR,
        DATA_DIR,
        CAMPUS_KB_DIR,
        VECTOR_STORE_DIR,
        CAMPUS_INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
