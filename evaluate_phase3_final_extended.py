"""Held-out Phase 3 comparison of PCC and four residual compensation models."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eval_phase3_final import load_dataset_provenance
from phase4_inverse_kinematics import (
    CANONICAL_MODEL_PATH,
    CANONICAL_X_SCALER_PATH,
    CANONICAL_Y_SCALER_PATH,
    Phase3NNCorrector,
)
from synthetic_validation_common import error_metrics, pcc_xyz, save_metric_table


DATASET = Path("dataset_out/dataset_3.npz")
OUT_DIR = Path("outputs/phase3_forward_compensation")
LOSS_CURVE = Path("phase3_last_16_03_2026/loss_curve.png")


def save_plots(out_dir: Path, pcc_position: np.ndarray, reference: np.ndarray,
               corrected_nn: np.ndarray, errors: dict[str, np.ndarray], seed: int) -> None:
    norm = {name: np.linalg.norm(error, axis=1) for name, error in errors.items()}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(norm["PCC"].max(), norm["PCC + NN"].max()), 46)
    ax.hist(norm["PCC"], bins=bins, alpha=0.6, label="PCC")
    ax.hist(norm["PCC + NN"], bins=bins, alpha=0.6, label="PCC + NN")
    ax.set(xlabel="Cartesian error norm [mm]", ylabel="Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist_pcc_vs_nn.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(list(norm.values()), showfliers=False)
    ax.set_xticks(np.arange(1, len(norm) + 1), list(norm))
    ax.set(ylabel="Cartesian error norm [mm]")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "model_error_boxplot.png", dpi=220)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(reference), size=min(600, len(reference)), replace=False)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for points, label, color in [
        (reference, "Synthetic reference", "black"),
        (pcc_position, "PCC", "#bd6546"),
        (corrected_nn, "PCC + NN", "#257a77"),
    ]:
        ax.scatter(*points[indices].T, s=7, alpha=0.55, label=label, color=color)
    ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "reference_pcc_nn_3d.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(3)
    pcc_boxes = ax.boxplot([np.abs(errors["PCC"][:, i]) for i in range(3)],
                           positions=positions - 0.17, widths=0.28,
                           showfliers=False, patch_artist=True)
    nn_boxes = ax.boxplot([np.abs(errors["PCC + NN"][:, i]) for i in range(3)],
                          positions=positions + 0.17, widths=0.28,
                          showfliers=False, patch_artist=True)
    for group, color in [(pcc_boxes, "#bd6546"), (nn_boxes, "#257a77")]:
        for item in group["boxes"]:
            item.set_facecolor(color)
    ax.set_xticks(positions, ["X", "Y", "Z"])
    ax.set(ylabel="Absolute axis error [mm]")
    ax.legend([pcc_boxes["boxes"][0], nn_boxes["boxes"][0]], ["PCC", "PCC + NN"])
    fig.tight_layout()
    fig.savefig(out_dir / "axis_errors.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for path in (DATASET, CANONICAL_MODEL_PATH, CANONICAL_X_SCALER_PATH,
                 CANONICAL_Y_SCALER_PATH, LOSS_CURVE):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(DATASET) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
    if X.ndim != 2 or Y.shape != X.shape or X.shape[1] != 3:
        raise ValueError("Canonical NPZ must contain matching (N, 3) X and Y arrays")

    seed, test_fraction, val_fraction = 42, 0.15, 0.15
    trainval_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_fraction,
                                              random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(trainval_idx,
                                        test_size=val_fraction / (1 - test_fraction),
                                        random_state=seed, shuffle=True)
    artifact_metrics_path = CANONICAL_MODEL_PATH.with_name("metrics.json")
    if artifact_metrics_path.is_file():
        artifact_metrics = json.loads(artifact_metrics_path.read_text(encoding="utf-8"))
        expected = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
        if artifact_metrics.get("seed") != seed or artifact_metrics.get("split_sizes") != expected:
            raise ValueError("Canonical NN provenance does not match the evaluation split")

    X_train, Y_train, X_test, Y_test = X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
    regressors = {
        "PCC + Linear Regression": LinearRegression(),
        "PCC + Polynomial Ridge": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False), StandardScaler(), Ridge(alpha=1.0)
        ),
        "PCC + KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    }
    predictions = {"PCC": np.zeros_like(Y_test)}
    for name, model in regressors.items():
        model.fit(X_train, Y_train)
        predictions[name] = model.predict(X_test)
    nn = Phase3NNCorrector(CANONICAL_MODEL_PATH, CANONICAL_X_SCALER_PATH,
                           CANONICAL_Y_SCALER_PATH, device=args.device)
    predictions["PCC + NN"] = nn.predict_delta(X_test)

    errors = {name: pred - Y_test for name, pred in predictions.items()}
    rows = [{"method": name, **error_metrics(error)} for name, error in errors.items()]
    table = save_metric_table(rows, args.out_dir, "phase3_metrics",
                              "Phase 3 held-out forward-position error (mm).",
                              "tab:phase3_extended")
    pcc_position = pcc_xyz(X_test)
    reference = pcc_position + Y_test
    save_plots(args.out_dir, pcc_position, reference,
               pcc_position + predictions["PCC + NN"], errors, seed)
    shutil.copy2(LOSS_CURVE, args.out_dir / "loss_curve.png")

    boundary = (np.max(X_test, axis=1) >= 9.0) & ((X_test > 1e-9).sum(axis=1) == 2)
    boundary_rows = []
    if boundary.any():
        boundary_rows = [{"method": name, **error_metrics(err[boundary])}
                         for name, err in errors.items()]
        save_metric_table(boundary_rows, args.out_dir, "phase3_boundary_metrics",
                          "Phase 3 large-bend test subset (mm).", "tab:phase3_extended_boundary")

    summary = {
        "reference_model": "synthetic reference model (perturbed PCC; no physical robot data)",
        "dataset": str(DATASET),
        "model": str(CANONICAL_MODEL_PATH),
        "x_scaler": str(CANONICAL_X_SCALER_PATH),
        "y_scaler": str(CANONICAL_Y_SCALER_PATH),
        "seed": seed,
        "split": {"test_fraction": test_fraction, "validation_fraction": val_fraction,
                  "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "regressors_fit_on": "train only; preprocessing is inside each fitted pipeline",
        "nn": "canonical pretrained residual NN; no retraining",
        "error_definition": "PCC + predicted residual - synthetic reference position",
        "dataset_provenance": load_dataset_provenance(DATASET),
        "metrics": rows,
        "boundary_rule": "max(dl) >= 9 mm and exactly two active tendons",
        "boundary_count": int(boundary.sum()),
        "boundary_metrics": boundary_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Boundary test points: {boundary.sum()}")
    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
