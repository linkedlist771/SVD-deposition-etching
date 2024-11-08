from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


if __name__ == "__main__":
    print(ROOT)
