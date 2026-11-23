from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "NAB_dataset"
SERIES_DIR = RAW / "data" / "realAWSCloudwatch"
LABELS = RAW / "labels" / "combined_labels.json"
WINDOWS = RAW / "labels" / "combined_windows.json"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

STEP = "5min"
TRAIN_FRAC = 0.7
HORIZON = 12