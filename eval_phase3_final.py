# Phase 3 final evaluation (no GUI)
# - single NPZ dataset
# - compares: PCC / PCC+PolyDeg3 / PCC+NN
# - exports tables, plots, summary.json

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn


# -----------------------
# Config
# -----------------------
@dataclass
class Cfg:
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15  # kept for split consistency

    # Dataset: X (dl1..dl3), Y = (REAL - PCC)
    npz_path: Path = Path("dataset_out/dataset_phase2_synth_new.npz")

    # NN artifacts
    nn_dir: Path = Path("phase3_version_3")
    nn_model_pt: Path = Path("phase3_version_3/nn_model.pt")
    x_scaler_pkl: Path = Path("phase3_version_3/x_scaler.pkl")
    y_scaler_pkl: Path = Path("phase3_version_3/y_scaler.pkl")

    # Output directory
    out_dir: Path = Path("phase3_final")

    # Boundary threshold
    boundary_thr: float = 9.0
    eps_activity: float = 1e-9

    # Max samples in 3D scatter
    scatter_n: int = 6000

    # PolyDeg3 settings
    poly_degree: int = 3
    ridge_alpha: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utils / Metrics
# -----------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vec_norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=1))


def metrics_from_err(err_vec: np.ndarray) -> dict:
    # err_vec: (N,3) residual (comp - real)
    n = vec_norm(err_vec)
    return {
        "MAE_norm": float(np.mean(np.abs(n))),
        "RMSE_norm": float(np.sqrt(np.mean(n ** 2))),
        "MAX_norm": float(np.max(n)),
        "MAE_X": float(np.mean(np.abs(err_vec[:, 0]))),
        "MAE_Y": float(np.mean(np.abs(err_vec[:, 1]))),
        "MAE_Z": float(np.mean(np.abs(err_vec[:, 2]))),
        "RMSE_X": float(np.sqrt(np.mean(err_vec[:, 0] ** 2))),
        "RMSE_Y": float(np.sqrt(np.mean(err_vec[:, 1] ** 2))),
        "RMSE_Z": float(np.sqrt(np.mean(err_vec[:, 2] ** 2))),
        "MAX_abs_X": float(np.max(np.abs(err_vec[:, 0]))),
        "MAX_abs_Y": float(np.max(np.abs(err_vec[:, 1]))),
        "MAX_abs_Z": float(np.max(np.abs(err_vec[:, 2]))),
        "N": int(err_vec.shape[0]),
    }


def save_table_csv_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str):
    df.to_csv(csv_path, index=False)

    # Minimal LaTeX table (booktabs)
    tex = []
    tex.append(r"\begin{table}[h!]")
    tex.append(r"\centering")
    tex.append(r"\caption{" + caption + r"}")
    tex.append(r"\label{" + label + r"}")
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(r"Metóda & MAE$_{\|\cdot\|}$ [mm] & RMSE$_{\|\cdot\|}$ [mm] & MAX$_{\|\cdot\|}$ [mm] \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        tex.append(
            f"{row['method']} & {row['MAE_norm']:.6f} & {row['RMSE_norm']:.6f} & {row['MAX_norm']:.6f} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex_path.write_text("\n".join(tex), encoding="utf-8")


# -----------------------
# NN (must match training)
# -----------------------
def get_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, hidden_sizes: tuple[int, ...] = (128, 64), activation: str = "relu"):
        super().__init__()
        act = get_activation(activation)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def nn_predict_dx(
    dl3: np.ndarray,
    model_pt: Path,
    x_scaler_pkl: Path,
    y_scaler_pkl: Path,
    device: str,
) -> np.ndarray:
    # Predict (REAL - PCC)
    x_scaler = joblib.load(x_scaler_pkl)
    y_scaler = joblib.load(y_scaler_pkl)

    ckpt = torch.load(model_pt, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    hidden_sizes = tuple(cfg_dict.get("hidden_sizes", (128, 64)))
    activation = str(cfg_dict.get("activation", "relu"))
    use_activity_mask = bool(cfg_dict.get("use_activity_mask", True))
    eps_activity = float(cfg_dict.get("eps_activity", 1e-9))
    in_dim = int(cfg_dict.get("in_dim", 6))

    dl = np.asarray(dl3, dtype=np.float64)
    if dl.ndim != 2 or dl.shape[1] != 3:
        raise ValueError(f"dl must be (N,3), got {dl.shape}")

    # Optional activity mask
    if use_activity_mask:
        A = (np.abs(dl) > eps_activity).astype(np.float64)
        feats = np.hstack([dl, A])
    else:
        feats = dl

    if feats.shape[1] != in_dim:
        raise ValueError(f"in_dim mismatch: {feats.shape[1]} vs {in_dim}")

    feats_n = x_scaler.transform(feats).astype(np.float32)

    model = FeedForwardNN(in_dim=in_dim, hidden_sizes=hidden_sizes, activation=activation).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_n = model(torch.from_numpy(feats_n).to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_n)
    return pred.astype(np.float64)


# -----------------------
# PolyDeg3 regression
# -----------------------
def train_polydeg3(X_train: np.ndarray, Y_train: np.ndarray, degree: int, alpha: float) -> Pipeline:
    # poly -> scaler -> ridge
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha)),
    ])
    model.fit(X_train, Y_train)
    return model


# -----------------------
# Plots
# -----------------------
def plot_hist(errs: dict[str, np.ndarray], out_path: Path, title: str):
    # Histogram of ||error||
    plt.figure()
    for name, e in errs.items():
        n = vec_norm(e)
        plt.hist(n, bins=60, alpha=0.5, label=name)
    plt.xlabel("||error|| [mm]")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter3d(errs: dict[str, np.ndarray], out_path: Path, title: str, nmax: int, seed: int):
    # 3D scatter of (dX,dY,dZ)
    rng = np.random.default_rng(seed)

    plt.figure()
    ax = plt.axes(projection="3d")  # type: ignore

    for name, e in errs.items():
        N = e.shape[0]
        s = e if N <= nmax else e[rng.choice(N, size=nmax, replace=False)]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, alpha=0.35, label=name)

    ax.set_xlabel("dX [mm]")
    ax.set_ylabel("dY [mm]")
    ax.set_zlabel("dZ [mm]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    cfg = Cfg()
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {cfg.npz_path.resolve()}")

    for p in [cfg.nn_model_pt, cfg.x_scaler_pkl, cfg.y_scaler_pkl]:
        if not p.exists():
            raise FileNotFoundError(f"Missing NN artifact: {p.resolve()}")

    data = np.load(cfg.npz_path)
    if "X" not in data or "Y" not in data:
        raise KeyError("NPZ must contain 'X' and 'Y'")

    X_all = data["X"].astype(np.float64)
    Y_all = data["Y"].astype(np.float64)

    if X_all.shape[1] != 3 or Y_all.shape[1] != 3:
        raise ValueError("Expected shape (N,3)")

    # Split: test -> train/val
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_all, Y_all, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
    )
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_rel, random_state=cfg.seed, shuffle=True
    )

    # 1) PCC baseline: residual = -Y
    err_test_pcc = -Y_test

    # 2) PolyDeg3 compensation
    poly = train_polydeg3(X_train, Y_train, degree=cfg.poly_degree, alpha=cfg.ridge_alpha)
    pred_test_poly = poly.predict(X_test).astype(np.float64)
    err_test_poly = -Y_test + pred_test_poly

    # 3) NN compensation
    pred_test_nn = nn_predict_dx(
        dl3=X_test,
        model_pt=cfg.nn_model_pt,
        x_scaler_pkl=cfg.x_scaler_pkl,
        y_scaler_pkl=cfg.y_scaler_pkl,
        device=cfg.device,
    )
    err_test_nn = -Y_test + pred_test_nn

    # Tables (TEST)
    rows_test = []
    for name, err in [
        ("PCC", err_test_pcc),
        ("PCC + PolyDeg3", err_test_poly),
        ("PCC + NN", err_test_nn),
    ]:
        rows_test.append({"method": name, **metrics_from_err(err)})

    df_test = pd.DataFrame(rows_test)
    df_test_compact = df_test[["method", "MAE_norm", "RMSE_norm", "MAX_norm"]].copy()

    save_table_csv_tex(
        df_test_compact,
        cfg.out_dir / "comparison_test.csv",
        cfg.out_dir / "comparison_test.tex",
        caption="Test set comparison.",
        label="tab:phase3_test_comparison",
    )

    # Boundary subset: max(dl) >= thr
    dlmax = np.max(X_test, axis=1)
    boundary_mask = dlmax >= cfg.boundary_thr

    active_cnt = (np.abs(X_test) > cfg.eps_activity).sum(axis=1)
    boundary_mask = boundary_mask & (active_cnt == 2)

    if int(boundary_mask.sum()) < 200:
        boundary_mask = dlmax >= cfg.boundary_thr

    err_b_pcc = err_test_pcc[boundary_mask]
    err_b_poly = err_test_poly[boundary_mask]
    err_b_nn = err_test_nn[boundary_mask]

    rows_b = []
    for name, err in [
        ("PCC", err_b_pcc),
        ("PCC + PolyDeg3", err_b_poly),
        ("PCC + NN", err_b_nn),
    ]:
        rows_b.append({"method": name, **metrics_from_err(err)})

    df_b = pd.DataFrame(rows_b)
    df_b_compact = df_b[["method", "MAE_norm", "RMSE_norm", "MAX_norm"]].copy()

    save_table_csv_tex(
        df_b_compact,
        cfg.out_dir / "comparison_boundary.csv",
        cfg.out_dir / "comparison_boundary.tex",
        caption=f"Boundary subset (max(dl) ≥ {cfg.boundary_thr:.1f} mm).",
        label="tab:phase3_boundary_comparison",
    )

    # Plots
    errs_test = {
        "PCC": err_test_pcc,
        "PCC+PolyDeg3": err_test_poly,
        "PCC+NN": err_test_nn,
    }
    errs_boundary = {
        "PCC": err_b_pcc,
        "PCC+PolyDeg3": err_b_poly,
        "PCC+NN": err_b_nn,
    }

    plot_hist(errs_test, cfg.out_dir / "errors_hist_test.png", "Error norm (TEST)")
    plot_hist(errs_boundary, cfg.out_dir / "errors_hist_boundary.png", "Error norm (BOUNDARY)")

    plot_scatter3d(errs_test, cfg.out_dir / "error_scatter3d_test.png", "Residuals XYZ (TEST)", cfg.scatter_n, cfg.seed)
    plot_scatter3d(errs_boundary, cfg.out_dir / "error_scatter3d_boundary.png", "Residuals XYZ (BOUNDARY)", cfg.scatter_n, cfg.seed + 1)

    # JSON summary
    summary = {
        "dataset": str(cfg.npz_path),
        "splits": {"seed": cfg.seed, "test_size": cfg.test_size, "val_size": cfg.val_size},
        "boundary_thr": cfg.boundary_thr,
        "test_table": df_test_compact.to_dict(orient="records"),
        "boundary_table": df_b_compact.to_dict(orient="records"),
        "nn_artifacts": {
            "model_pt": str(cfg.nn_model_pt),
            "x_scaler": str(cfg.x_scaler_pkl),
            "y_scaler": str(cfg.y_scaler_pkl),
            "device": cfg.device,
        },
        "polydeg3": {"degree": cfg.poly_degree, "ridge_alpha": cfg.ridge_alpha},
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== TEST ===")
    print(df_test_compact.to_string(index=False))
    print("\n=== BOUNDARY ===")
    print(df_b_compact.to_string(index=False))
    print(f"\nSaved to: {cfg.out_dir.resolve()}")


if __name__ == "__main__":
    main()