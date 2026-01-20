import numpy as np
import pandas as pd
from pathlib import Path

import pcc_model as pcc
import real_model as real


# Dataset generation parameters
SEED = 42                  # fixed seed for reproducibility
N_SAMPLES = 30000          # dataset size

DL_MIN = 0.0               # minimum tendon shortening [mm]
DL_MAX = 10.0              # maximum tendon shortening [mm]
ENFORCE_MAX_2 = True       # physical constraint: max 2 active tendons

OUT_DIR = Path("dataset_out")
OUT_CSV = OUT_DIR / "dataset.csv"
OUT_STATS = OUT_DIR / "stats.txt"


# Configure real model parameters if supported
try:
    real.alpha_per_m = 3.5
    real.beta_rad_per_m = 3.5
    real.offset = np.array([0.5, 0.5, 0.3], dtype=float)
    real.sigma_noise = 0.5
    real.theta_max = np.radians(95)
except Exception:
    pass


# Sample tendon shortenings with max 2 active tendons
def sample_dls(n: int, dl_min: float, dl_max: float, rng: np.random.Generator) -> np.ndarray:
    dls = np.zeros((n, 3), dtype=float)

    # mix of 1-tend and 2-tend activations
    cases = rng.choice([1, 2], size=n, p=[0.4, 0.6])

    for i in range(n):
        if cases[i] == 1:
            j = rng.integers(0, 3)
            dls[i, j] = rng.uniform(dl_min, dl_max)
        else:
            j1, j2 = rng.choice(3, size=2, replace=False)
            dls[i, j1] = rng.uniform(dl_min, dl_max)
            dls[i, j2] = rng.uniform(dl_min, dl_max)

    return dls


# Tip position from ideal PCC model
def pcc_tip_xyz(dl1, dl2, dl3):
    X, Y, Z, _theta = pcc.pcc_shape(dl1, dl2, dl3, n_points=60)
    return float(X[-1]), float(Y[-1]), float(Z[-1])


# Tip position from simulated real model
def real_tip_xyz(dl1, dl2, dl3, enforce_max2: bool):
    try:
        x, y, z, _theta = real.real_forward(
            dl1, dl2, dl3, enforce_limit=enforce_max2
        )
    except TypeError:
        x, y, z, _theta = real.real_forward(dl1, dl2, dl3)

    return float(x), float(y), float(z)


# Generate PCC vs REAL error dataset
def generate_dataset():
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


# Dataset statistics
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
    text.append("Input distribution bins [mm]: [0,2,4,6,8,10]")
    bins = [0, 2, 4, 6, 8, 10]
    for col in ["dl1", "dl2", "dl3"]:
        hist = np.histogram(df[col].values, bins=bins)[0]
        text.append(f"  {col}: {hist.tolist()}")

    return "\n".join(text)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(OUT_CSV, index=False)

    stats = make_stats_text(df)
    OUT_STATS.write_text(stats, encoding="utf-8")

    print(f"Saved dataset: {OUT_CSV} (rows={len(df)})")
    print(f"Saved stats  : {OUT_STATS}")


if __name__ == "__main__":
    main()
