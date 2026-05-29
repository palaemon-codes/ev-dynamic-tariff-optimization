from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ACN_DIR = PROJECT_ROOT / "data" / "raw" / "acn"
RAW_URBANEV_DIR = PROJECT_ROOT / "data" / "raw" / "urbanev"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

BASELINE_TARIFF = 15.0
DEFAULT_SEED = 42
