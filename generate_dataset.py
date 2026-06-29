import json
import numpy as np
import pandas as pd
from pathlib import Path

import pcc_model as pcc
import real_model as real


# Dataset parameters
SEED = 42                  # RNG seed
N_SAMPLES = 20000         # number of samples

DL_MIN = 0.0               # min shortening [mm]
DL_MAX = 10.0              # max shortening [mm]
ENFORCE_MAX_2 = True       # max 2 active tendons

OUT_DIR = Path("dataset_out")
OUT_CSV = OUT_DIR / "dataset_3.csv"
OUT_STATS = OUT_DIR / "stats_3.txt"
OUT_PROVENANCE = OUT_DIR / "dataset_3_provenance.json"

REAL_MODEL_PARAMS = {
    "alpha_per_m": 3.5,
    "beta_rad_per_m": 3.5,
    "offset": [0.5, 0.5, 0.3],
    "sigma_noise": 0.5,
    "theta_max_deg": 95.0,
}


# Configure real model (if available)
def apply_real_model_params():
    real.alpha_per_m = REAL_MODEL_PARAMS["alpha_per_m"]
    real.beta_rad_per_m = REAL_MODEL_PARAMS["beta_rad_per_m"]
    real.offset = np.array(REAL_MODEL_PARAMS["offset"], dtype=float)
    real.sigma_noise = REAL_MODEL_PARAMS["sigma_noise"]
    real.theta_max = np.radians(REAL_MODEL_PARAMS["theta_max_deg"])


apply_real_model_params()


def gaussian_noise_floor_norm() -> float:
    # Expected RMSE of the 3D Euclidean residual norm for iid per-axis Gaussian noise.
    return float(np.sqrt(3.0) * REAL_MODEL_PARAMS["sigma_noise"])


# Sample dl with ≤2 active tendons
def sample_dls(n: int, dl_min: float, dl_max: float, rng: np.random.Generator,
               p_single: float = 0.4, p_boundary: float = 0.30, k_boundary: float = 4.0,
               eps: float = 1e-9) -> np.ndarray:
    """
    Modes:
      single: 1 active tendon
      pair  : 2 active tendons
    Amplitude:
      boundary-heavy or uniform
    """
    dls = np.zeros((n, 3), dtype=float)

    is_single = rng.random(n) < p_single
    is_boundary = rng.random(n) < p_boundary

    def sample_amp(boundary: bool) -> float:
        if not boundary:
            return float(rng.uniform(dl_min, dl_max))
        u = rng.random()
        return float(dl_min + (dl_max - dl_min) * (1.0 - (u ** k_boundary)))

    single_modes = np.array([0, 1, 2])
    pair_modes = np.array([(0, 1), (0, 2), (1, 2)])

    for i in range(n):
        if is_single[i]:
            j = int(rng.choice(single_modes))
            dls[i, j] = sample_amp(bool(is_boundary[i]))
        else:
            j1, j2 = pair_modes[int(rng.integers(0, 3))]
            dls[i, j1] = sample_amp(bool(is_boundary[i]))
            dls[i, j2] = sample_amp(bool(is_boundary[i]))

    # safety check
    active = (np.abs(dls) > eps).sum(axis=1)
    assert np.all(active <= 2), "More than 2 active tendons"

    return dls


# PCC tip position
def pcc_tip_xyz(dl1, dl2, dl3):
    X, Y, Z, _theta = pcc.pcc_shape(dl1, dl2, dl3, n_points=60)
    return float(X[-1]), float(Y[-1]), float(Z[-1])


# Real model tip position
def real_tip_xyz(dl1, dl2, dl3, enforce_max2: bool):
    try:
        x, y, z, _theta = real.real_forward(
            dl1, dl2, dl3, enforce_limit=enforce_max2
        )
    except TypeError:
        x, y, z, _theta = real.real_forward(dl1, dl2, dl3)

    return float(x), float(y), float(z)


# Generate dataset: dl → (PCC, REAL, error)
def generate_dataset():
    apply_real_model_params()
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dls = sample_dls(N_SAMPLES, DL_MIN, DL_MAX, rng)

    rows = []
    for dl1, dl2, dl3 in dls:
        xp, yp, zp = pcc_tip_xyz(dl1, dl2, dl3)
        xr, yr, zr = real_tip_xyz(dl1, dl2, dl3, ENFORCE_MAX_2)

        dx = xr - xp
        dy = yr - yp
        dz = zr - zp
        dnorm = float(np.sqrt(dx*dx + dy*dy + dz*dz))

        rows.append({
            "dl1": float(dl1), "dl2": float(dl2), "dl3": float(dl3),

            "X_pcc": xp, "Y_pcc": yp, "Z_pcc": zp,
            "X_real": xr, "Y_real": yr, "Z_real": zr,

            "dX": dx, "dY": dy, "dZ": dz,
            "dXYZ": dnorm
        })

    return pd.DataFrame(rows)


# Basic dataset statistics
def make_stats_text(df: pd.DataFrame) -> str:
    active = (df[["dl1", "dl2", "dl3"]].abs() > 1e-9).sum(axis=1)
    active_counts = active.value_counts().sort_index()

    err = df["dXYZ"]
    dx, dy, dz = df["dX"], df["dY"], df["dZ"]

    text = []
    text.append("=== DATASET STATS ===")
    text.append(f"N samples: {len(df)}")
    text.append("")

    text.append("Input limits:")
    text.append(f"  dl in [{DL_MIN}, {DL_MAX}] mm")
    text.append(f"Seed: {SEED}")
    text.append(f"Estimated Gaussian noise floor RMSE_norm [mm]: {gaussian_noise_floor_norm():.4f}")

    text.append("Active tendons:")
    for k in [0, 1, 2, 3]:
        text.append(f"  {k}: {int(active_counts.get(k, 0))}")

    text.append("")
    text.append("Error magnitude dXYZ [mm]:")
    text.append(f"  mean: {err.mean():.4f}")
    text.append(f"  std : {err.std():.4f}")
    text.append(f"  p95 : {err.quantile(0.95):.4f}")
    text.append(f"  max : {err.max():.4f}")

    text.append("")
    text.append("Axis errors [mm]:")
    text.append(f"  dX: mean={dx.mean():.4f}, std={dx.std():.4f}, max_abs={dx.abs().max():.4f}")
    text.append(f"  dY: mean={dy.mean():.4f}, std={dy.std():.4f}, max_abs={dy.abs().max():.4f}")
    text.append(f"  dZ: mean={dz.mean():.4f}, std={dz.std():.4f}, max_abs={dz.abs().max():.4f}")

    text.append("")
    text.append("Input histogram bins [mm]: [0,2,4,6,8,10]")
    bins = [0, 2, 4, 6, 8, 10]
    for col in ["dl1", "dl2", "dl3"]:
        hist = np.histogram(df[col].values, bins=bins)[0]
        text.append(f"  {col}: {hist.tolist()}")

    return "\n".join(text)


def make_provenance(df: pd.DataFrame) -> dict:
    return {
        "dataset_csv": str(OUT_CSV),
        "seed": SEED,
        "n_samples": int(len(df)),
        "dl_min": DL_MIN,
        "dl_max": DL_MAX,
        "enforce_max_2": ENFORCE_MAX_2,
        "real_model_parameters": REAL_MODEL_PARAMS,
        "estimated_gaussian_noise_floor_rmse_norm_mm": gaussian_noise_floor_norm(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(OUT_CSV, index=False)

    stats = make_stats_text(df)
    OUT_STATS.write_text(stats, encoding="utf-8")
    OUT_PROVENANCE.write_text(
        json.dumps(make_provenance(df), indent=2),
        encoding="utf-8",
    )

    print(f"Saved dataset: {OUT_CSV} (rows={len(df)})")
    print(f"Saved stats  : {OUT_STATS}")
    print(f"Saved provenance: {OUT_PROVENANCE}")


if __name__ == "__main__":
    main()
