import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor


# Config
SEED = 42

DATASET_CSV = Path("dataset_out") / "dataset.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios: train/val/test with val=0.15 and test=0.15 of total
TEST_FRAC = 0.15
VAL_FRAC_FROM_REMAIN = 0.1764705882  # 0.15 / 0.85

# Input features and labels
X_COLS = ["dl1", "dl2", "dl3"]
PCC_COLS = ["X_pcc", "Y_pcc", "Z_pcc"]
REAL_COLS = ["X_real", "Y_real", "Z_real"]
Y_COLS = ["dX", "dY", "dZ"]  # targets are errors (REAL - PCC) to be compensated

# Polynomial regression degree
POLY_DEGREE = 3

# KNN hyperparameters
KNN_K = 25
KNN_WEIGHTS = "distance"


# Metrics helpers
def vec_norm(v: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norm for vectors shaped (N,3)."""
    return np.sqrt(np.sum(v * v, axis=1))


def metrics_report(name: str, err_vec: np.ndarray) -> dict:
    """
    Compute MAE/RMSE/MAX on vector norm and on each axis for error vectors (N,3) in mm.
    """
    norm = vec_norm(err_vec)
    out = {
        "model": name,
        "MAE_norm": float(np.mean(np.abs(norm))),
        "RMSE_norm": float(np.sqrt(np.mean(norm ** 2))),
        "MAX_norm": float(np.max(norm)),
        "MAE_X": float(np.mean(np.abs(err_vec[:, 0]))),
        "MAE_Y": float(np.mean(np.abs(err_vec[:, 1]))),
        "MAE_Z": float(np.mean(np.abs(err_vec[:, 2]))),
        "RMSE_X": float(np.sqrt(np.mean(err_vec[:, 0] ** 2))),
        "RMSE_Y": float(np.sqrt(np.mean(err_vec[:, 1] ** 2))),
        "RMSE_Z": float(np.sqrt(np.mean(err_vec[:, 2] ** 2))),
        "MAX_abs_X": float(np.max(np.abs(err_vec[:, 0]))),
        "MAX_abs_Y": float(np.max(np.abs(err_vec[:, 1]))),
        "MAX_abs_Z": float(np.max(np.abs(err_vec[:, 2]))),
    }
    return out


def print_table(rows: list[dict]):
    # Print a compact summary focused on norm-based metrics for quick comparison
    cols = ["model", "MAE_norm", "RMSE_norm", "MAX_norm"]
    header = " | ".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{r[c]:12.6f}" if c != "model" else f"{r[c]:>12}" for c in cols))


# Load & split
def load_dataset(path: Path) -> pd.DataFrame:
    # Fail early if dataset is missing or has wrong schema
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    for c in X_COLS + PCC_COLS + REAL_COLS + Y_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in dataset.")
    return df


def make_splits(df: pd.DataFrame):
    # X = tendon shortenings, y = error vector (REAL - PCC)
    X = df[X_COLS].to_numpy(dtype=float)
    y = df[Y_COLS].to_numpy(dtype=float)

    # Keep PCC and REAL xyz to evaluate baseline vs compensated accuracy
    pcc = df[PCC_COLS].to_numpy(dtype=float)
    real = df[REAL_COLS].to_numpy(dtype=float)

    # Split out test first so final metrics stay untouched by model selection
    X_rem, X_test, y_rem, y_test, pcc_rem, pcc_test, real_rem, real_test = train_test_split(
        X, y, pcc, real, test_size=TEST_FRAC, random_state=SEED, shuffle=True
    )

    # Split remaining into train/val (val becomes 15% of total)
    X_train, X_val, y_train, y_val, pcc_train, pcc_val, real_train, real_val = train_test_split(
        X_rem, y_rem, pcc_rem, real_rem,
        test_size=VAL_FRAC_FROM_REMAIN, random_state=SEED, shuffle=True
    )

    return (X_train, y_train, pcc_train, real_train), (X_val, y_val, pcc_val, real_val), (X_test, y_test, pcc_test, real_test)


# Models
def build_models():
    # All models predict y=[dX,dY,dZ] from X=[dl1,dl2,dl3]
    models = {}

    # Linear baseline: fast and interpretable
    models["Linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])

    # Polynomial + Ridge: captures nonlinearity while keeping coefficients stable
    models["PolyDeg3"] = Pipeline([
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])

    # KNN: nonparametric baseline that can fit local behavior well
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=KNN_K, weights=KNN_WEIGHTS)),
    ])

    return models


# Evaluation logic
def evaluate_model(model, X, y_true, pcc_xyz, real_xyz, name: str):
    """
    Baseline error: PCC - REAL
    Compensated error: (PCC + ML_predicted_error) - REAL
    """
    baseline_err = pcc_xyz - real_xyz

    # Model predicts the correction vector (REAL - PCC) in mm
    y_pred = model.predict(X)
    comp_xyz = pcc_xyz + y_pred
    comp_err = comp_xyz - real_xyz

    report_baseline = metrics_report(f"{name} :: PCC baseline", baseline_err)
    report_comp = metrics_report(f"{name} :: PCC+ML", comp_err)

    return report_baseline, report_comp


def main():
    df = load_dataset(DATASET_CSV)
    (X_train, y_train, pcc_train, real_train), (X_val, y_val, pcc_val, real_val), (X_test, y_test, pcc_test, real_test) = make_splits(df)

    # Quick sanity prints to confirm dataset and split sizes
    print("Dataset loaded:", DATASET_CSV)
    print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print("Features:", X_COLS, "Targets:", Y_COLS)
    print("")

    models = build_models()

    all_results_val = []
    all_results_test = []

    # Train + evaluate each model the same way for fair comparison
    for name, model in models.items():
        print(f"=== Training: {name} ===")
        model.fit(X_train, y_train)

        # Validation metrics for model selection
        b_val, c_val = evaluate_model(model, X_val, y_val, pcc_val, real_val, name)
        all_results_val.extend([b_val, c_val])

        # Test metrics as final unbiased estimate
        b_test, c_test = evaluate_model(model, X_test, y_test, pcc_test, real_test, name)
        all_results_test.extend([b_test, c_test])

        # Persist trained pipeline (scaler + model) as a single artifact
        out_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, out_path)
        print(f"Saved: {out_path}")
        print("")

    # Human-readable summary on validation and test
    print("\n===== VALIDATION SUMMARY (norm metrics) =====")
    print_table(all_results_val)

    print("\n===== TEST SUMMARY (norm metrics) =====")
    print_table(all_results_test)

    # Save full metrics tables for later analysis / thesis figures
    val_df = pd.DataFrame(all_results_val)
    test_df = pd.DataFrame(all_results_test)
    val_df.to_csv(MODELS_DIR / "metrics_val.csv", index=False)
    test_df.to_csv(MODELS_DIR / "metrics_test.csv", index=False)

    print("\nSaved metrics tables:")
    print(" ", MODELS_DIR / "metrics_val.csv")
    print(" ", MODELS_DIR / "metrics_test.csv")

    # Pick the best compensated model on validation set by RMSE_norm
    val_comp = val_df[val_df["model"].str.contains("PCC\\+ML", regex=True)].copy()
    best_row = val_comp.loc[val_comp["RMSE_norm"].idxmin()]
    print("\nBest on VAL (by RMSE_norm, compensated):")
    print(best_row[["model", "MAE_norm", "RMSE_norm", "MAX_norm"]].to_string())


if __name__ == "__main__":
    main()
