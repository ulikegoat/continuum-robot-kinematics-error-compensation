"""Read-only adapters for validated synthetic and Phase 6 v2 artifacts.

Only endpoint measurements are available for the physical robot. Displayed
backbones are analytical PCC geometry, never measured physical curves.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from continuum_robot.common.paths import ROOT, PHASE3_MODEL
from continuum_robot.kinematics import pcc_model, synthetic_reference
from continuum_robot.phase3.train_nn import FeedForwardNN

V2_RESULTS = ROOT / "results" / "final" / "phase6_review_v2"
V2_MODELS = ROOT / "artifacts" / "phase6_review_v2"
EXPERIMENTS = ("random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3")
PHYSICAL_MODELS = ("PCC", "KNN", "FFNN", "Ridge3", "Ridge2", "Linear")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curve(parts) -> np.ndarray:
    return np.column_stack(parts[:3]).astype(float)


class SyntheticAdapter:
    """Deterministic synthetic reference and the saved canonical Phase 3 NN."""

    def __init__(self):
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.warning = ""
        required = (PHASE3_MODEL / "nn_model.pt", PHASE3_MODEL / "x_scaler.pkl", PHASE3_MODEL / "y_scaler.pkl")
        missing = [p.name for p in required if not p.exists()]
        if missing:
            self.warning = "Phase 3 NN unavailable: " + ", ".join(missing)
            return
        try:
            checkpoint = torch.load(required[0], map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})
            self.use_activity_mask = bool(config.get("use_activity_mask", True))
            self.activity_epsilon = float(config.get("eps_activity", 1e-9))
            self.model = FeedForwardNN(in_dim=int(config.get("in_dim", 3)),
                                       hidden_sizes=tuple(config.get("hidden_sizes", (64, 64))),
                                       activation=config.get("activation", "relu"))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.x_scaler = joblib.load(required[1])
            self.y_scaler = joblib.load(required[2])
        except Exception as exc:
            self.model = None
            self.warning = f"Phase 3 NN could not be loaded: {exc}"

    def evaluate(self, shortening_mm: np.ndarray) -> dict:
        dl = np.asarray(shortening_mm, float).reshape(3)
        pcc_curve = _curve(pcc_model.pcc_shape(*dl, n_points=80))
        reference_curve = _curve(synthetic_reference.real_shape(*dl, n_points=80, enforce_limit=True))
        # real_forward adds a random measurement-noise draw. The GUI uses its
        # deterministic expectation, including the model's endpoint bias.
        reference = reference_curve[-1] + synthetic_reference.offset
        corrected = None
        if self.model is not None:
            features = dl.reshape(1, 3)
            if self.use_activity_mask:
                features = np.hstack((features, (np.abs(features) > self.activity_epsilon).astype(float)))
            scaled = self.x_scaler.transform(features).astype(np.float32)
            with torch.no_grad():
                residual = self.y_scaler.inverse_transform(self.model(torch.from_numpy(scaled)).numpy())[0]
            corrected = pcc_curve[-1] + residual
        return {"pcc_curve": pcc_curve, "reference_curve": reference_curve,
                "pcc": pcc_curve[-1], "reference": reference, "corrected": corrected,
                "theta_deg": float(np.degrees(pcc_model.pcc_phi_kappa_theta(*dl)[2])),
                "reference_label": "Deterministic synthetic reference",
                "frame_label": "Synthetic PCC coordinates [mm]", "warning": self.warning}


class PhysicalAdapter:
    """A single Phase 6 v2 experiment, preserving its registration and scalers."""

    def __init__(self, experiment: str):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown physical experiment: {experiment}")
        self.experiment = experiment
        self.result_dir = V2_RESULTS / experiment
        self.model_dir = V2_MODELS / experiment
        needed = (self.result_dir / "registration.json", self.result_dir / "predictions.csv",
                  self.result_dir / "metrics.csv", self.result_dir / "provenance.json",
                  self.model_dir / "metadata.json", self.model_dir / "input_scaler.joblib",
                  self.model_dir / "residual_scaler.joblib")
        missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError("Phase 6 v2 artifacts unavailable: " + ", ".join(missing))
        provenance = json.loads((self.result_dir / "provenance.json").read_text(encoding="utf-8"))
        metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        registration = json.loads((self.result_dir / "registration.json").read_text(encoding="utf-8"))
        if metadata["version"] != provenance["version"] or metadata["experiment"] != experiment:
            raise ValueError("Phase 6 v2 metadata/provenance mismatch")
        if _sha(self.model_dir / "metadata.json") != provenance["model_metadata_sha256"]:
            raise ValueError("Phase 6 v2 metadata changed since evaluation")
        if _sha(self.result_dir / "registration.json") != metadata["registration_sha256"]:
            raise ValueError("Phase 6 v2 registration changed since training")
        if _sha(self.result_dir / "predictions.csv") != provenance["predictions_sha256"]:
            raise ValueError("Phase 6 v2 predictions changed since evaluation")
        if _sha(self.result_dir / "metrics.csv") != provenance["metrics_sha256"]:
            raise ValueError("Phase 6 v2 metrics changed since evaluation")
        for filename in ("input_scaler.joblib", "residual_scaler.joblib"):
            if _sha(self.model_dir / filename) != provenance["model_file_sha256"][filename]:
                raise ValueError(f"Phase 6 v2 scaler changed since evaluation: {filename}")
        self.metadata = metadata
        self.registration = registration["selected"]
        self.order = self.registration["order"]
        self.rotation = np.asarray(self.registration["row_rotation"], float)
        self.translation = np.asarray(self.registration["translation_mm"], float)
        self.predictions = pd.read_csv(self.result_dir / "predictions.csv")
        self.metrics = pd.read_csv(self.result_dir / "metrics.csv")
        self.x_scaler = joblib.load(self.model_dir / "input_scaler.joblib")
        self.y_scaler = joblib.load(self.model_dir / "residual_scaler.joblib")
        self._loaded = {}
        self._by_command = {
            tuple(int(v) for v in row[1:4]): index
            for index, row in enumerate(self.predictions[["row_id", "l1", "l2", "l3"]].itertuples(index=False, name=None))
        }

    def _load_model(self, name: str):
        if name in self._loaded:
            return self._loaded[name]
        if name == "PCC":
            return None
        path = self.model_dir / ("FFNN.pt" if name == "FFNN" else f"{name}.joblib")
        if not path.exists():
            raise FileNotFoundError(f"Selected {name} model missing: {path.relative_to(ROOT)}")
        expected = json.loads((self.result_dir / "provenance.json").read_text(encoding="utf-8"))["model_file_sha256"].get(path.name)
        if expected and _sha(path) != expected:
            raise ValueError(f"Selected {name} model changed since evaluation")
        if name == "FFNN":
            architecture = tuple(self.metadata["selection"]["FFNN"]["architecture"])
            model = FeedForwardNN(3, 3, architecture, "relu")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
        else:
            model = joblib.load(path)
        self._loaded[name] = model
        return model

    def evaluate(self, shortening_mm: np.ndarray, model_name: str) -> dict:
        if model_name not in PHYSICAL_MODELS:
            raise ValueError(f"Unknown compensation model: {model_name}")
        dl = np.asarray(shortening_mm, float).reshape(3)
        local_curve = _curve(pcc_model.pcc_shape(*dl[self.order], n_points=80))
        registered_curve = local_curve @ self.rotation + self.translation
        pcc_tip = registered_curve[-1]
        corrected = pcc_tip.copy()
        if model_name != "PCC":
            model = self._load_model(model_name)
            scaled = self.x_scaler.transform(dl.reshape(1, 3)).astype(np.float32)
            if model_name == "FFNN":
                with torch.no_grad():
                    delta_scaled = model(torch.from_numpy(scaled)).numpy()
            else:
                delta_scaled = model.predict(scaled)
            corrected = pcc_tip + self.y_scaler.inverse_transform(delta_scaled)[0]
        # Exact observations belong to the selected experiment's TEST split.
        # Slider positions between sampled commands have no observed endpoint here.
        commands = dl * 10
        key = tuple(int(round(v)) for v in commands)
        index = self._by_command.get(key) if np.allclose(commands, key, atol=1e-8, rtol=0) else None
        observed = None if index is None else self.predictions.iloc[index][["x_real", "y_real", "z_real"]].to_numpy(float)
        return {"pcc_curve": registered_curve, "reference_curve": None,
                "pcc": pcc_tip, "reference": observed, "corrected": corrected,
                "theta_deg": float(np.degrees(pcc_model.pcc_phi_kappa_theta(*dl[self.order])[2])),
                "reference_label": "Measured test endpoint" if observed is not None else "No selected-test measurement",
                "frame_label": "Assumed measurement coordinates [mm]",
                "warning": "Empirical train-only registration; sensor frame is not independently calibrated"}

    def cloud(self, model_name: str, limit: int = 4500):
        frame = self.predictions
        if len(frame) > limit:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]
        positions = frame[["x_real", "y_real", "z_real"]].to_numpy(float)
        error = frame[f"e_{model_name}"].to_numpy(float)
        return positions, error

    def test_command(self, index: int) -> np.ndarray:
        row = self.predictions.iloc[index]
        return row[["l1", "l2", "l3"]].to_numpy(float) / 10
