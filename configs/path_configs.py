from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

if __name__ == "__main__":
    print(ROOT)
