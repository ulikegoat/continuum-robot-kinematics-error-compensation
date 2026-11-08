"""Phase 6 physical-data pipeline (legacy technical module name: phase5).

A verified external calibration is required for model scoring.

Run with ``python -m continuum_robot.phase5.pipeline --help``.
Original measurements are immutable; row_id always refers to the source row.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from continuum_robot.common.paths import ROOT
from continuum_robot.common.synthetic_validation_common import error_metrics, pcc_xyz
from continuum_robot.kinematics import pcc_model
from continuum_robot.phase3.train_nn import FeedForwardNN
from continuum_robot.visualization.thesis_plot_style import (
    apply_thesis_style, equal_3d_axes, light_grid, save_figure,
)

DATA = ROOT / "data" / "real"
RESULTS = ROOT / "results" / "final" / "phase5_real_compensation"
MODELS = ROOT / "artifacts" / "phase5_models"
INPUTS = ["l1", "l2", "l3"]
XYZ = ["x_real", "y_real", "z_real"]
PCC = ["x_pcc", "y_pcc", "z_pcc"]
RESIDUAL = ["dx", "dy", "dz"]
METHODS = ["PCC", "Linear", "Ridge2", "Ridge3", "FFNN"]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 6:
        raise ValueError("Source must have six numeric columns without a header")
    frame.columns = INPUTS + XYZ
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


def analyze(source: Path, out: Path = DATA, position_scale: float = 1.0):
    out.mkdir(parents=True, exist_ok=True)
    before = digest(source)
    frame = load_source(source)
    arr = frame[INPUTS + XYZ].to_numpy(float)
    finite = np.isfinite(arr).all(axis=1)
    u = frame[INPUTS].to_numpy(float)
    valid = finite & (u >= 0).all(1) & (u <= 100).all(1)
    valid &= np.isclose(u, np.round(u), atol=1e-8).all(1) & ((u != 0).sum(1) <= 2)
    frame["is_valid"] = valid
    # A local predictor uses only available one-command (0.1 mm) neighbours.
    # Flags are descriptive QA, not verified bad measurements.
    lookup = {}
    for row in np.flatnonzero(valid):
        lookup.setdefault(tuple(u[row].astype(int)), []).append(row)
    local_deviation = np.full(len(frame), np.nan)
    neighbour_count = np.zeros(len(frame), dtype=int)
    xyz = frame[XYZ].to_numpy(float) * position_scale
    edges = []
    for key, rows in lookup.items():
        neighbours = []
        for axis in range(3):
            for step in (-1, 1):
                other = list(key); other[axis] += step
                indices = lookup.get(tuple(other), [])
                if indices:
                    neighbours.append(xyz[indices].mean(0))
                    if step == 1:
                        for i in rows:
                            for j in indices:
                                edges.append((i, j, axis, float(np.linalg.norm(xyz[i] - xyz[j]))))
        if len(neighbours) >= 2:
            center = np.median(neighbours, axis=0)
            local_deviation[rows] = np.linalg.norm(xyz[rows] - center, axis=1)
            neighbour_count[rows] = len(neighbours)
    values = local_deviation[np.isfinite(local_deviation)]
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    threshold = max(1.0, median + 6 * 1.4826 * mad)
    frame["local_deviation_mm_assuming_position_scale"] = local_deviation
    frame["neighbour_count"] = neighbour_count
    frame["is_outlier"] = valid & (local_deviation > threshold)
    frame[["row_id"] + INPUTS + XYZ].to_csv(out / "real_dataset_raw.csv", index=False)
    frame.to_csv(out / "real_dataset_flagged.csv", index=False)
    edge_frame = pd.DataFrame(edges, columns=["row_id_a", "row_id_b", "axis", "jump_mm_assuming_position_scale"])
    edge_frame.to_csv(out / "local_neighbour_jumps.csv", index=False)
    expected = set()
    for a, b in itertools.product(range(101), repeat=2):
        expected.update(((a, b, 0), (a, 0, b), (0, a, b)))
    expected.discard((0, 0, 0))
    missing = sorted(expected - set(lookup))
    pd.DataFrame(missing, columns=INPUTS).to_csv(out / "missing_configurations.csv", index=False)
    frame[frame.duplicated(INPUTS, keep=False)].to_csv(out / "duplicate_configurations.csv", index=False)
    report = {
        "source": str(source), "source_sha256": before, "rows": len(frame),
        "unique_configurations": len(frame[INPUTS].drop_duplicates()),
        "exact_duplicate_extra_rows": int(frame.duplicated(INPUTS + XYZ).sum()),
        "duplicate_configuration_extra_rows": int(frame.duplicated(INPUTS).sum()),
        "nan_by_column": frame[INPUTS + XYZ].isna().sum().astype(int).to_dict(),
        "inf_by_column": dict(zip(INPUTS + XYZ, np.isinf(arr).sum(0).astype(int).tolist())),
        "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
        "ranges_source_units": {c: ([float(frame.loc[np.isfinite(frame[c]), c].min()), float(frame.loc[np.isfinite(frame[c]), c].max())] if np.isfinite(frame[c]).any() else [None, None]) for c in INPUTS + XYZ},
        "missing_nonzero_plane_grid_configurations": len(missing),
        "zero_configuration_present": (0, 0, 0) in lookup,
        "expected_grid_assumption": "Union of three integer 0..100 two-tendon planes, excluding origin",
        "command_to_mm": 0.1, "position_to_mm_assumption": position_scale,
        "flagged_rows": int(frame.is_outlier.sum()),
        "outlier_rule": "distance from componentwise median of immediate neighbours > max(1 mm, median + 6*1.4826*MAD)",
        "outlier_threshold_mm": threshold, "outlier_median_mm": median, "outlier_mad_mm": mad,
        "outlier_caution": "Global descriptive flags may include mechanical jumps. Clean sensitivity analysis is conditional on these flags; full test set remains primary.",
        "active_tendon_counts": {str(k): int(v) for k, v in pd.Series((u != 0).sum(1)).value_counts().items()},
        "neighbour_jump_percentiles_mm": {str(q): float(np.percentile(edge_frame.iloc[:, -1], q)) for q in [50, 95, 99, 100]} if len(edges) else {},
    }
    write_json(out / "dataset_analysis.json", report)
    if digest(source) != before:
        raise RuntimeError("Source changed during analysis")
    apply_thesis_style()
    figures = RESULTS / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*xyz[valid].T, s=1, alpha=.35, label="Measured positions")
    ax.scatter(*xyz[frame.is_outlier].T, s=6, color="#B45F45", label="QA flags")
    ax.set(xlabel="X [assumed mm]", ylabel="Y [assumed mm]", zlabel="Z [assumed mm]", title="Physical measurements: workspace and QA flags")
    equal_3d_axes(ax, xyz[valid]); ax.legend(); save_figure(fig, figures / "measured_workspace.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(values, bins=100); axes[0].axvline(threshold, color="#B45F45")
    axes[0].set(xlabel="Local deviation [assumed mm]", ylabel="Count", title="Descriptive outlier flags")
    if len(edges): axes[1].hist(edge_frame.iloc[:, -1], bins=100)
    axes[1].set(xlabel="Neighbour position jump [assumed mm]", ylabel="Count", title="0.1 mm command steps")
    for ax in axes: light_grid(ax)
    save_figure(fig, figures / "measurement_quality.png")
    return report


def read_calibration(path: Path | None) -> dict:
    if path is None or not path.exists():
        raise ValueError("Verified calibration required: supply --calibration data/real/calibration.json. See calibration_required.md.")
    cal = json.loads(path.read_text(encoding="utf-8"))
    if cal.get("verified") is not True or not cal.get("provenance"):
        raise ValueError("Calibration requires verified=true and documented provenance")
    r = np.asarray(cal["pcc_to_measurement_rotation"], float)
    t = np.asarray(cal["pcc_to_measurement_translation_mm"], float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("Invalid rotation/translation shape or nonfinite values")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
        raise ValueError("Coordinate transform must be orthogonal (no fitted scale/shear)")
    if any(type(v) is not int for v in cal["dataset_columns_in_pcc_tendon_order"]) or sorted(cal["dataset_columns_in_pcc_tendon_order"]) != [0, 1, 2]:
        raise ValueError("Tendon order must be a permutation of [0,1,2]")
    if not np.isfinite(float(cal["measurement_units_to_mm"])) or float(cal["measurement_units_to_mm"]) <= 0:
        raise ValueError("measurement_units_to_mm must be positive")
    if not np.isclose(cal["backbone_length_mm"], pcc_model.L) or not np.isclose(cal["routing_radius_mm"], pcc_model.d):
        raise ValueError("Confirmed physical geometry differs from existing PCC; model parameterisation must be resolved first")
    if cal.get("position_marker") != "tip":
        raise ValueError("Existing PCC predicts tip position; marker offset must be resolved before comparison")
    return cal


def register_pcc(commands: np.ndarray, cal: dict) -> np.ndarray:
    local = pcc_xyz(commands[:, cal["dataset_columns_in_pcc_tendon_order"]] / 10)
    return local @ np.asarray(cal["pcc_to_measurement_rotation"]).T + np.asarray(cal["pcc_to_measurement_translation_mm"])


def diagnostic(data: Path = DATA):
    frame = pd.read_csv(data / "real_dataset_flagged.csv")
    valid = frame[frame.is_valid].copy()
    commands = valid[INPUTS].to_numpy(float); xyz = valid[XYZ].to_numpy(float)
    raw = pcc_xyz(commands / 10)
    for i, col in enumerate(["x_pcc_local", "y_pcc_local", "z_pcc_local"]): valid[col] = raw[:, i]
    valid.to_csv(data / "real_vs_pcc_unregistered.csv", index=False)
    fits = []
    for order in itertools.permutations(range(3)):
        model = pcc_xyz(commands[:, order] / 10)
        a = model - model.mean(0); b = xyz - xyz.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        translation = xyz.mean(0) - model.mean(0) @ rotation
        fits.append({"tendon_order": list(order), "diagnostic_rotation_column_convention": rotation.T.tolist(),
                     "diagnostic_translation_mm_assumption": translation.tolist(),
                     "global_fit_rmse_mm_assumption": error_metrics(model @ rotation + translation - xyz)["RMSE_norm"]})
    report = {"status": "unregistered; numerical diagnostics are NOT physical PCC error metrics",
              "position_unit_assumption": "mm; not independently verified", "pcc_geometry": {"L_mm": pcc_model.L, "d_mm": pcc_model.d},
              "pcc_zero": pcc_xyz(np.zeros((1, 3)))[0].tolist(),
              "near_zero_measurements": valid[valid[INPUTS].sum(axis=1) == 1][["row_id"] + INPUTS + XYZ].to_dict("records"),
              "unaligned_coordinate_distance_rmse": error_metrics(raw - xyz)["RMSE_norm"],
              "rigid_fits_diagnostic_only": fits,
              "warning": "Global fitting confounds geometry errors with coordinate alignment and uses all measurements. These transforms must not be used for held-out model evaluation."}
    write_json(RESULTS / "coordinate_diagnostics.json", report)
    write_json(RESULTS / "phase5_status.json", {"status": "blocked_on_verified_calibration", "analysis_complete": True,
        "real_models_trained": False, "RQ1_RQ5": "Pending verified coordinate transform, geometry, units and marker/tendon convention"})
    return report


def generate_predictions(calibration: Path, data: Path = DATA):
    cal = read_calibration(calibration)
    analysis = json.loads((data / "dataset_analysis.json").read_text())
    if not np.isclose(analysis["position_to_mm_assumption"], cal["measurement_units_to_mm"]):
        raise ValueError("Rerun analyze with --position-scale matching calibration before using QA flags")
    frame = pd.read_csv(data / "real_dataset_flagged.csv")
    frame = frame[frame.is_valid].copy()
    if frame.duplicated(INPUTS).any():
        raise ValueError("Repeated commands require grouped splits; resolve duplicate configurations before training")
    prediction = register_pcc(frame[INPUTS].to_numpy(float), cal)
    frame[XYZ] = frame[XYZ] * cal["measurement_units_to_mm"]
    for i, col in enumerate(PCC): frame[col] = prediction[:, i]
    frame[RESIDUAL] = frame[XYZ].to_numpy() - prediction
    frame["e_pcc"] = np.linalg.norm(frame[RESIDUAL].to_numpy(), axis=1)
    frame.to_csv(data / "real_vs_pcc.csv", index=False)
    write_json(data / "prediction_provenance.json", {"calibration": cal, "calibration_sha256": digest(calibration),
        "input_sha256": digest(data / "real_dataset_flagged.csv"), "prediction_sha256": digest(data / "real_vs_pcc.csv"),
        "residual_sign": "measurement minus registered PCC", "evaluation_frame": "measurement frame, mm"})
    write_json(RESULTS / "phase5_status.json", {"status": "calibrated_predictions_ready", "analysis_complete": True,
        "calibration_sha256": digest(calibration), "physical_experiment_results": "pending"})
    write_json(RESULTS / "registered_pcc_metrics.json", {
        "all_valid": error_metrics(prediction - frame[XYZ].to_numpy()),
        "unflagged_descriptive_subset": error_metrics(prediction[~frame.is_outlier] - frame.loc[~frame.is_outlier, XYZ].to_numpy()) if (~frame.is_outlier).any() else None,
        "calibration_sha256": digest(calibration), "units": "mm",
    })
    apply_thesis_style()
    figures = RESULTS / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for zero, ax in enumerate(axes):
        active = [a for a in range(3) if a != zero]; mask = frame[INPUTS[zero]] == 0
        sc = ax.scatter(frame.loc[mask, INPUTS[active[0]]]/10, frame.loc[mask, INPUTS[active[1]]]/10,
                        c=frame.loc[mask, "e_pcc"], vmin=0, vmax=frame.e_pcc.max(), cmap="viridis", s=3)
        ax.set(xlabel=f"Tendon {active[0]+1} [mm]", ylabel=f"Tendon {active[1]+1} [mm]", xlim=(0,10), ylim=(0,10))
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes.tolist(), label="Registered PCC error [mm]")
    save_figure(fig, figures / "pcc_full_plane_error_maps.png")
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*frame[XYZ].to_numpy().T, c=frame.e_pcc, s=2, cmap="viridis")
    equal_3d_axes(ax, frame[XYZ].to_numpy()); ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]", title="Measured workspace: registered PCC error")
    fig.colorbar(sc, ax=ax, shrink=.65, label="Registered PCC error [mm]")
    save_figure(fig, figures / "pcc_workspace_error.png")
    return frame


def split_specs(frame: pd.DataFrame, seed: int):
    ids = frame.row_id.to_numpy(int)
    trainval, test = train_test_split(ids, test_size=.15, random_state=seed)
    train, val = train_test_split(trainval, test_size=.15/.85, random_state=seed)
    yield "random", train, val, test, {"kind": "random interpolation", "ratios": [0.7, 0.15, 0.15]}
    commands = frame[INPUTS].to_numpy(int)
    for step in (2, 5, 10):
        mask = (commands % step == 0).all(axis=1)
        train, val = train_test_split(ids[mask], test_size=.15, random_state=seed)
        yield f"sparse_{step}", train, val, ids[~mask], {"kind": "sparse interpolation", "step_mm": step / 10, "validation": "15% of coarse-grid pool"}
    for zero in range(3):
        active = [a for a in range(3) if a != zero]
        region = (commands[:, zero] == 0) & ((commands[:, active] >= 40) & (commands[:, active] <= 60)).all(1)
        train, val = train_test_split(ids[~region], test_size=.15, random_state=seed)
        yield f"region_{zero+1}", train, val, ids[region], {"kind": "contiguous region holdout", "zero_tendon": zero+1, "bounds_mm": [4, 6]}


def prepare(data: Path = DATA, seed: int = 42):
    prediction_file = data / "real_vs_pcc.csv"
    basis = prediction_file if prediction_file.exists() else data / "real_dataset_flagged.csv"
    frame = pd.read_csv(basis)
    frame = frame[frame.is_valid].copy()
    if frame.duplicated(INPUTS).any():
        raise ValueError("Repeated configurations require grouped splitting")
    if prediction_file.exists():
        validate_predictions(data)
        frame[["row_id"] + INPUTS + RESIDUAL + ["is_outlier"]].to_csv(data / "residual_dataset.csv", index=False)
    counts = []
    for name, train, val, test, spec in split_specs(frame, seed):
        if min(len(train), len(val), len(test)) == 0:
            raise ValueError(f"Empty split: {name}")
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
            raise RuntimeError("Overlapping split")
        write_json(RESULTS / "splits" / f"{name}.json", {"seed": seed, "spec": spec, "basis": basis.name, "dataset_sha256": digest(basis),
            "train": train.tolist(), "validation": val.tolist(), "test": test.tolist()})
        counts.append({"experiment": name, "seed": seed, "train": len(train), "validation": len(val), "test": len(test)})
    pd.DataFrame(counts).to_csv(RESULTS / "split_summary.csv", index=False)
    print(f"Saved seven splits; residual targets {'ready' if prediction_file.exists() else 'pending verified calibration'}", flush=True)


def validate_predictions(data: Path):
    prediction = data / "real_vs_pcc.csv"
    provenance_file = data / "prediction_provenance.json"
    if not prediction.exists() or not provenance_file.exists():
        raise ValueError("Calibrated predictions missing; run predict with a verified calibration first")
    provenance = json.loads(provenance_file.read_text())
    if provenance.get("registration_method") != "train_only_empirical" and provenance["calibration"].get("verified") is not True:
        raise ValueError("Prediction calibration is unverified")
    if digest(prediction) != provenance["prediction_sha256"] or digest(DATA / "real_dataset_flagged.csv") != provenance["input_sha256"]:
        raise ValueError("Predictions or analysis changed; regenerate predictions and splits")


def arrays(frame: pd.DataFrame, ids):
    selected = frame.loc[ids]
    return selected[INPUTS].to_numpy(float) / 10, selected[RESIDUAL].to_numpy(float)


def train_experiment(name: str, data: Path = DATA, epochs: int = 300, seed: int = 42):
    validate_predictions(data)
    split = json.loads((RESULTS / "splits" / f"{name}.json").read_text())
    expected_basis = data / split["basis"] if (data / split["basis"]).exists() else DATA / split["basis"]
    if split["dataset_sha256"] != digest(expected_basis):
        raise ValueError("Dataset changed after splitting")
    frame = pd.read_csv(data / "real_vs_pcc.csv").set_index("row_id")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for variant in ("all", "clean_train"):
        train_ids = np.array(split["train"], int); val_ids = np.array(split["validation"], int)
        if variant == "clean_train":
            train_ids = train_ids[~frame.loc[train_ids].is_outlier.to_numpy(bool)]
            val_ids = val_ids[~frame.loc[val_ids].is_outlier.to_numpy(bool)]
        if min(len(train_ids), len(val_ids)) < 2:
            raise ValueError("Too few training/validation rows after QA filtering")
        x, y = arrays(frame, train_ids); xv, yv = arrays(frame, val_ids)
        folder = MODELS / name / variant; folder.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed); torch.manual_seed(seed)
        sx = StandardScaler().fit(x); sy = StandardScaler().fit(y)
        joblib.dump(sx, folder / "x_scaler.pkl"); joblib.dump(sy, folder / "y_scaler.pkl")
        xs = sx.transform(x); ys = sy.transform(y)
        xt = torch.tensor(xs, dtype=torch.float32); yt = torch.tensor(ys, dtype=torch.float32)
        xvt = torch.tensor(sx.transform(xv), dtype=torch.float32)
        yvt = torch.tensor(sy.transform(yv), dtype=torch.float32)
        choices = {}
        for method in ("Linear", "Ridge2", "Ridge3"):
            if method == "Linear": model = LinearRegression().fit(xs, ys)
            else:
                degree = int(method[-1]); best_score = float("inf")
                for alpha in (1e-4, .01, 1., 100.):
                    candidate = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
                    candidate.fit(xs, ys)
                    score = error_metrics(sy.inverse_transform(candidate.predict(sx.transform(xv))) - yv)["RMSE_norm"]
                    if score < best_score:
                        model, best_score, best_alpha = candidate, score, alpha
                choices[method] = {"degree": degree, "alpha": best_alpha, "validation_RMSE_mm": best_score}
            joblib.dump(model, folder / f"{method}.pkl")
        model = FeedForwardNN(3, 3, (215, 68), "relu")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-6)
        history = []; best = float("inf"); stale = 0; best_epoch = 0; best_state = None
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            model.train(); permutation = torch.randperm(len(xt), generator=generator); total = 0.
            for indices in permutation.split(256):
                optimizer.zero_grad(); loss = ((model(xt[indices]) - yt[indices]) ** 2).mean()
                loss.backward(); optimizer.step(); total += float(loss.detach()) * len(indices)
            model.eval()
            with torch.no_grad(): val_loss = float(((model(xvt) - yvt) ** 2).mean())
            history.append({"epoch": epoch + 1, "train_MSE_scaled": total/len(xt), "validation_MSE_scaled": val_loss})
            if val_loss < best - 1e-6:
                best = val_loss; stale = 0; best_epoch = epoch + 1
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else: stale += 1
            if stale >= 30: break
        torch.save(best_state, folder / "FFNN.pt")
        pd.DataFrame(history).to_csv(folder / "loss_history.csv", index=False)
        metadata = {"seed": seed, "split": split, "actual_train": train_ids.tolist(), "actual_validation": val_ids.tolist(),
            "variant": variant, "dataset_sha256": digest(data / "real_vs_pcc.csv"),
            "calibration_sha256": json.loads((data / "prediction_provenance.json").read_text()).get("calibration_sha256"),
            "scalers_fit_on": "actual_train only", "ridge_selection": choices,
            "nn": {"in_dim": 3, "out_dim": 3, "hidden_sizes": [215, 68], "activation": "relu", "lr": .001,
                   "weight_decay": 1e-6, "batch_size": 256, "max_epochs": epochs, "patience": 30, "best_epoch": best_epoch},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}}
        write_json(folder / "metadata.json", metadata)
        print(f"trained {name}/{variant}: train={len(x)}, validation={len(xv)}, best epoch={best_epoch}", flush=True)


def evaluate_experiment(name: str, data: Path = DATA):
    validate_predictions(data)
    frame = pd.read_csv(data / "real_vs_pcc.csv").set_index("row_id")
    output = RESULTS / name; output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("all", "clean_train"):
        folder = MODELS / name / variant
        meta = json.loads((folder / "metadata.json").read_text())
        if meta["dataset_sha256"] != digest(data / "real_vs_pcc.csv"):
            raise ValueError("Evaluation dataset differs from trained model dataset")
        ids = meta["split"]["test"]; xt, target = arrays(frame, ids)
        sx = joblib.load(folder / "x_scaler.pkl"); sy = joblib.load(folder / "y_scaler.pkl")
        scaled = sx.transform(xt)
        model = FeedForwardNN(3, 3, tuple(meta["nn"]["hidden_sizes"]), "relu")
        model.load_state_dict(torch.load(folder / "FFNN.pt", map_location="cpu", weights_only=True)); model.eval()
        predictions = {"PCC": np.zeros_like(target)}
        for method in ("Linear", "Ridge2", "Ridge3"):
            predictions[method] = sy.inverse_transform(joblib.load(folder / f"{method}.pkl").predict(scaled))
        with torch.no_grad(): predictions["FFNN"] = sy.inverse_transform(model(torch.tensor(scaled, dtype=torch.float32)).numpy())
        result = frame.loc[ids].copy()
        for method, correction in predictions.items():
            pred = result[PCC].to_numpy() + correction
            error = correction - target
            result[f"e_{method}"] = np.linalg.norm(error, axis=1)
            for a, axis in enumerate("xyz"):
                result[f"{axis}_{method}"] = pred[:, a]
                result[f"d{axis}_{method}"] = correction[:, a]
            for subset, mask in (("full_test", np.ones(len(ids), bool)), ("unflagged_test", ~result.is_outlier.to_numpy(bool))):
                if mask.any(): rows.append({"experiment": name, "variant": variant, "subset": subset, "model": method, **error_metrics(error[mask])})
        result.reset_index().to_csv(output / f"predictions_{variant}.csv", index=False)
    metrics = pd.DataFrame(rows); metrics.to_csv(output / "metrics.csv", index=False)
    compact = metrics[["variant", "subset", "model", "N", "MAE_norm", "RMSE_norm", "Median_norm", "P95_norm", "MAX_norm", "RMSE_X", "RMSE_Y", "RMSE_Z"]]
    (output / "metrics.tex").write_text(compact.to_latex(index=False, float_format="%.6f"), encoding="utf-8")
    write_json(output / "evaluation.json", {"mode": "loaded models; no retraining", "test_ids_identical_for_variants": True,
        "prediction_dataset": str(data / "real_vs_pcc.csv"),
        "dataset_sha256": digest(data / "real_vs_pcc.csv"), "units": "mm", "error": "registered PCC + predicted residual - measurement"})
    print(f"evaluated {name}: {len(ids)} test rows", flush=True)


def plot_experiment(name: str):
    apply_thesis_style(); output = RESULTS / name
    frame = pd.read_csv(output / "predictions_all.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        values = frame[f"e_{method}"].to_numpy()
        if method in ("PCC", "FFNN"): axes[0].hist(values, bins=60, alpha=.45, label=method)
        axes[1].plot(np.sort(values), np.arange(1, len(values)+1)/len(values), label=method)
    axes[0].set(xlabel="Position error [mm]", ylabel="Count", title=name)
    axes[1].set(xlabel="Position error [mm]", ylabel="CDF")
    for ax in axes: ax.legend(); light_grid(ax)
    save_figure(fig, output / "error_distribution.png")
    vmax = max(frame[f"e_{m}"].max() for m in METHODS)
    # Every plane/model shares the same axes and colour scale.
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), constrained_layout=True)
    for zero in range(3):
        active = [a for a in range(3) if a != zero]; mask = frame[INPUTS[zero]] == 0
        for j, method in enumerate(METHODS):
            ax = axes[zero, j]
            sc = ax.scatter(frame.loc[mask, INPUTS[active[0]]]/10, frame.loc[mask, INPUTS[active[1]]]/10,
                            c=frame.loc[mask, f"e_{method}"], vmin=0, vmax=vmax, cmap="viridis", s=5)
            ax.set(xlim=(0, 10), ylim=(0, 10), xlabel=f"Tendon {active[0]+1} [mm]", ylabel=f"Tendon {active[1]+1} [mm]", title=method)
            ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Test position error [mm]")
    save_figure(fig, output / "plane_error_maps.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, ax in zip("xyz", axes):
        actual = frame[f"{axis}_real"]
        for method in ("PCC", "Ridge3", "FFNN"): ax.scatter(actual, frame[f"{axis}_{method}"], s=3, alpha=.3, label=method)
        low, high = actual.min(), actual.max(); ax.plot([low, high], [low, high], "k--", lw=1)
        ax.set(xlabel=f"Measured {axis.upper()} [mm]", ylabel=f"Predicted {axis.upper()} [mm]"); light_grid(ax)
    axes[0].legend(); save_figure(fig, output / "measured_vs_predicted.png")
    for variant in ("all", "clean_train"):
        history = pd.read_csv(MODELS / name / variant / "loss_history.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.epoch, history.train_MSE_scaled, label="Train")
        ax.plot(history.epoch, history.validation_MSE_scaled, label="Validation")
        ax.set(xlabel="Epoch", ylabel="MSE (normalised residual)", yscale="log"); ax.legend(); light_grid(ax)
        save_figure(fig, output / f"loss_{variant}.png")
    if name.startswith("region"):
        zero = int(name[-1])-1; active = [a for a in range(3) if a != zero]
        meta = json.loads((MODELS / name / "all" / "metadata.json").read_text())
        full = pd.read_csv(DATA / "real_vs_pcc.csv").set_index("row_id")
        pool = full.loc[meta["actual_train"]]; pool = pool[pool[INPUTS[zero]] == 0]
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(pool[INPUTS[active[0]]]/10, pool[INPUTS[active[1]]]/10, s=2, label="Training")
        sc = ax.scatter(frame[INPUTS[active[0]]]/10, frame[INPUTS[active[1]]]/10, c=frame.e_FFNN, s=8, cmap="viridis")
        ax.set(xlabel=f"Tendon {active[0]+1} [mm]", ylabel=f"Tendon {active[1]+1} [mm]", xlim=(0,10), ylim=(0,10))
        ax.legend(); fig.colorbar(sc, ax=ax, label="FFNN holdout error [mm]"); save_figure(fig, output / "held_out_region.png")


def summarise():
    tables = [pd.read_csv(p) for p in sorted(RESULTS.glob("*/metrics.csv"))]
    required = {"random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3"}
    if {p.parent.name for p in RESULTS.glob("*/metrics.csv")} != required:
        raise ValueError("All seven experiments must be evaluated before final report")
    for name in required:
        record = json.loads((RESULTS / name / "evaluation.json").read_text())
        if record["dataset_sha256"] != digest(Path(record["prediction_dataset"])):
            raise ValueError(f"Stale evaluation: {name}")
    frame = pd.concat(tables, ignore_index=True); frame.to_csv(RESULTS / "all_metrics.csv", index=False)
    primary = frame[(frame.variant == "all") & (frame.subset == "full_test")]
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        selected = primary[(primary.model == method) & primary.experiment.str.startswith("sparse")].copy()
        selected["step"] = selected.experiment.str.split("_").str[-1].astype(int)/10
        selected = selected.sort_values("step")
        ax.plot(selected.step, selected.RMSE_norm, marker="o", label=method)
    ax.set(xlabel="Training grid spacing [mm]", ylabel="Test RMSE [mm]"); ax.legend(); light_grid(ax)
    save_figure(fig, RESULTS / "figures" / "sparse_grid_accuracy.png")
    baseline = primary[(primary.experiment == "random") & (primary.model == "PCC")].iloc[0]
    best = primary[(primary.experiment == "random") & (primary.model != "PCC")].sort_values("RMSE_norm").iloc[0]
    nn = primary[(primary.experiment == "random") & (primary.model == "FFNN")].iloc[0]
    registration = json.loads((DATA / "experiments" / "random" / "prediction_provenance.json").read_text()) if (DATA / "experiments" / "random" / "prediction_provenance.json").exists() else None
    calibration_description = "PCC was rigidly registered from training data only; tendon permutation selected on validation data. Measurement positions assumed mm. This is a data-identified baseline, not independently measured calibration." if registration else "All errors are in mm in the externally calibrated measurement frame."
    lines = ["# Phase 6 physical measurement results", "", calibration_description,
        "The six tendon permutations are geometrically symmetric after allowing arbitrary rigid rotations/reflections; their nearly identical validation scores do not identify the physical cable numbering. The selected permutation is an internal coordinate convention.",
        "Primary results retain all valid measurements, including QA flags. Clean sensitivity results are conditional on globally computed descriptive flags.", "",
        f"RQ1: Registered PCC random-test RMSE = {baseline.RMSE_norm:.6f}; MAE = {baseline.MAE_norm:.6f}; P95 = {baseline.P95_norm:.6f}.",
        f"RQ2: Best compensation ({best.model}) RMSE = {best.RMSE_norm:.6f}, reduction = {100*(1-best.RMSE_norm/baseline.RMSE_norm):.2f}%.",
        f"RQ3: FFNN RMSE = {nn.RMSE_norm:.6f}; Ridge3 RMSE = {primary[(primary.experiment == 'random') & (primary.model == 'Ridge3')].iloc[0].RMSE_norm:.6f}. Numerical differences alone do not establish statistical significance.",
        "RQ4: Sparse grid test RMSE:", ""]
    lines.append(primary[primary.experiment.str.startswith("sparse")][["experiment", "model", "N", "RMSE_norm"]].to_string(index=False))
    lines += ["", "RQ5: Contiguous region test RMSE (each plane):", "", primary[primary.experiment.str.startswith("region")][["experiment", "model", "N", "RMSE_norm"]].to_string(index=False),
        "", "Limitations: static positions, at most two active tendons, single seed, unknown repeatability and motion history; no physical IK validation and no evidence for three-tendon configurations."]
    (RESULTS / "research_answers.md").write_text("\n".join(lines), encoding="utf-8")
    missing_plots = [str(RESULTS / name / filename) for name in required
        for filename in ("error_distribution.png", "plane_error_maps.png", "measured_vs_predicted.png", "loss_all.png", "loss_clean_train.png")
        if not (RESULTS / name / filename).exists()]
    missing_plots += [str(RESULTS / name / "held_out_region.png") for name in required if name.startswith("region") and not (RESULTS / name / "held_out_region.png").exists()]
    write_json(RESULTS / "phase5_status.json", {"status": "complete" if not missing_plots else "evaluated_missing_figures", "real_models_trained": True,
        "experiments": sorted(primary.experiment.unique().tolist()), "missing_figures": missing_plots})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["analyze", "diagnostic", "predict", "prepare", "train", "evaluate", "plot", "run"])
    parser.add_argument("--source", type=Path, default=DATA / "Dataset-Actions-Positions.txt")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--experiment", default="all", choices=["all", "random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--position-scale", type=float, default=1.)
    args = parser.parse_args()
    if args.epochs < 1 or args.position_scale <= 0: parser.error("epochs and position-scale must be positive")
    stage = args.stage
    if stage in ("analyze", "run"):
        print(json.dumps(analyze(args.source, position_scale=args.position_scale), indent=2)); diagnostic()
    if stage == "diagnostic": print(json.dumps(diagnostic(), indent=2))
    if stage in ("predict", "run"):
        try: generate_predictions(args.calibration)
        except ValueError as exc: parser.exit(2, f"Phase 6 calibration gate: {exc}\n")
    if stage in ("prepare", "run"): prepare(seed=args.seed)
    names = [p.stem for p in sorted((RESULTS / "splits").glob("*.json"))] if args.experiment == "all" else [args.experiment]
    if stage in ("train", "evaluate", "plot", "run") and not names: parser.error("No splits; run prepare first")
    for name in names:
        if stage in ("train", "run"): train_experiment(name, epochs=args.epochs, seed=args.seed)
        if stage in ("evaluate", "run"): evaluate_experiment(name)
        if stage in ("plot", "run"): plot_experiment(name)
    if stage in ("plot", "run") and args.experiment == "all": summarise()


if __name__ == "__main__": main()
