"""Project constants - paths, thresholds, model version."""

from pathlib import Path

# Filesystem
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

# Model
MODEL_VERSION: str = "1.0.0"
LABEL_MAPPING: dict[str, str] = {"0": "NO_WORK_ORDER", "1": "WO"}

# Input validation - enforced before inference
MIN_READINGS: int = 5           # fewest acceptable timestamps per sample
TEMP_MIN: float = -50.0         # F - below this is a sensor fault
TEMP_MAX: float = 250.0         # F - above this is a sensor fault

# Feature extraction - normal HVAC operating band derived from EDA (~47-56 F)
TEMP_BAND_HIGH: float = 80.0
TEMP_BAND_LOW: float = 20.0

# Risk level thresholds - applied to prob(WO), independent of the predicted label
RISK_CRITICAL: float = 0.90
RISK_HIGH: float = 0.70
RISK_MEDIUM: float = 0.40
