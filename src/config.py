from pathlib import Path

# Automatically resolve the project root
PROJ_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJ_ROOT / "data"

# Checkpoint path
PROCESSED_CSV_PATH = DATA_DIR / "processed_decisions.csv"

# Base directory for training runs / checkpoints
RUNS_PATH = PROJ_ROOT / "runs"
