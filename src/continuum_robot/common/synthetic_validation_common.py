"""Shared reporting and synthetic-reference helpers for final validation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["MAE_norm", "RMSE_norm", "Median_norm", "P95_norm", "MAX_norm"]


def error_metrics(error_xyz: np.ndarray) -> dict:
    """Metrics for signed Cartesian error: predicted/reached minus reference/target."""
    error_xyz = np.asarray(error_xyz, dtype=np.float64)
    if error_xyz.ndim != 2 or error_xyz.shape[1] != 3 or len(error_xyz) == 0:
        raise ValueError("error_xyz must have shape (N, 3) with N > 0")
    norm = np.linalg.norm(error_xyz, axis=1)
    metrics = {
        "N": int(len(norm)),
        "MAE_norm": float(np.mean(norm)),
        "RMSE_norm": float(np.sqrt(np.mean(norm**2))),
        "Median_norm": float(np.median(norm)),
        "P95_norm": float(np.percentile(norm, 95)),
        "MAX_norm": float(np.max(norm)),
    }
    for axis, i in zip("XYZ", range(3)):
        values = error_xyz[:, i]
        metrics[f"MAE_{axis}"] = float(np.mean(np.abs(values)))
        metrics[f"RMSE_{axis}"] = float(np.sqrt(np.mean(values**2)))
        metrics[f"MAX_abs_{axis}"] = float(np.max(np.abs(values)))
    return metrics


def save_metric_table(rows: list[dict], out_dir: Path, stem: str, caption: str, label: str) -> pd.DataFrame:
    """Write the same compact result table as CSV and thesis-ready LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    leading = [c for c in df.columns if c not in METRIC_COLUMNS and c != "N"]
    columns = leading + ["N"] + METRIC_COLUMNS
    compact = df[columns]
    compact.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.tex").write_text(
        compact.to_latex(index=False, float_format="%.6f", caption=caption, label=label),
        encoding="utf-8",
    )
    return compact


def pcc_xyz(dls: np.ndarray) -> np.ndarray:
    from continuum_robot.kinematics import pcc_model as pcc

    dls = np.asarray(dls, dtype=np.float64).reshape(-1, 3)
    return np.array([pcc.pcc_forward(*map(float, dl))[:3] for dl in dls], dtype=np.float64)


def sample_commands(n: int, seed: int, dl_min: float = 0.0, dl_max: float = 10.0) -> np.ndarray:
    from continuum_robot.data.generate_dataset import sample_dls

    if n < 1 or not (0 <= dl_min < dl_max <= 10):
        raise ValueError("Require n >= 1 and 0 <= dl_min < dl_max <= 10")
    return sample_dls(n, dl_min, dl_max, np.random.default_rng(seed))


def synthetic_xyz(dls: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Evaluate the configured synthetic reference model outside any IK solver."""
    from continuum_robot.phase4.inverse_kinematics import configure_real_model

    if sigma < 0:
        raise ValueError("Noise sigma must be nonnegative")
    model = configure_real_model(sigma)
    # The synthetic reference forward function uses NumPy's global Gaussian RNG.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return np.array(
            [model.real_forward(*map(float, dl), enforce_limit=True)[:3] for dl in dls],
            dtype=np.float64,
        )
    finally:
        np.random.set_state(state)
