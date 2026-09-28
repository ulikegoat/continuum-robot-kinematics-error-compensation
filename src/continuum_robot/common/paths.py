"""Repository-relative locations shared by command-line modules."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FINAL_DATA = ROOT / "data" / "final"
LEGACY_DATA = ROOT / "data" / "legacy"
PHASE3_MODEL = ROOT / "artifacts" / "phase3_model"
LEGACY_MODELS = ROOT / "artifacts" / "legacy_models"
FINAL_RESULTS = ROOT / "results" / "final"
LEGACY_RESULTS = ROOT / "results" / "legacy"
FIGURES = ROOT / "figures"
