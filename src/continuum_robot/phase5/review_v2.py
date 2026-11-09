"""Phase 6 physical-data review protocol, version 2.

No work happens on import. ``run`` trains only when explicitly requested.
The previous Phase 6 pipeline and its outputs remain an archival comparator.
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from continuum_robot.common.paths import ROOT
from continuum_robot.common.synthetic_validation_common import error_metrics, pcc_xyz
from continuum_robot.phase3.train_nn import FeedForwardNN
from continuum_robot.kinematics import pcc_model
from continuum_robot.visualization.thesis_plot_style import apply_thesis_style, equal_3d_axes, light_grid, save_figure
from continuum_robot.phase5 import pipeline as legacy

VERSION = "phase6_review_v2_2026_11_09"
EXPERIMENTS = ("random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3")
METHODS = ("PCC", "Linear", "Ridge2", "Ridge3", "KNN", "FFNN")
DEFAULT_DATA = ROOT / "data" / "real"
DEFAULT_SPLITS = ROOT / "results" / "final" / "phase5_real_compensation" / "splits"
DEFAULT_OUTPUT = ROOT / "results" / "final" / "phase6_review_v2"
DEFAULT_MODELS = ROOT / "artifacts" / "phase6_review_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_inputs(data: Path, splits: Path, name: str):
    """Use original source rows and the exact archived split; reject stale inputs."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    source = data / "Dataset-Actions-Positions.txt"
    flagged_file = data / "real_dataset_flagged.csv"
    analysis = json.loads((data / "dataset_analysis.json").read_text(encoding="utf-8"))
    if sha(source) != analysis["source_sha256"]:
        raise ValueError("Source changed since analysis; stop and reanalyse explicitly")
    if float(analysis["position_to_mm_assumption"]) != 1.0:
        raise ValueError("Protocol assumes source XYZ in mm; document and implement any alternative unit first")
    split = json.loads((splits / f"{name}.json").read_text(encoding="utf-8"))
    if split["basis"] != flagged_file.name or split["dataset_sha256"] != sha(flagged_file):
        raise ValueError("Split does not match flagged source; recreate a versioned protocol explicitly")
    frame = pd.read_csv(flagged_file).set_index("row_id", drop=False)
    raw = legacy.load_source(source).set_index("row_id", drop=False)
    if not frame.index.is_unique or not raw.index.equals(frame.index):
        raise ValueError("Processed row IDs differ from the physical source")
    values = legacy.INPUTS + legacy.XYZ
    if not np.allclose(raw[values].to_numpy(float), frame[values].to_numpy(float), equal_nan=True, rtol=0, atol=0):
        raise ValueError("Processed measurements differ from the immutable source")
    frame = frame[frame.is_valid].copy()
    if frame.duplicated(legacy.INPUTS).any():
        raise ValueError("Repeated commands need grouped splits before model fitting")
    groups = [np.asarray(split[k], dtype=int) for k in ("train", "validation", "test")]
    if any(len(x) == 0 or len(np.unique(x)) != len(x) for x in groups):
        raise ValueError("Empty split or duplicate row ID")
    if any(set(groups[i]) & set(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Training/validation/test overlap")
    if set(np.concatenate(groups)) != set(frame.index):
        raise ValueError("Split must partition all valid physical measurements")
    return frame, split, groups, {"source_sha256": sha(source), "flagged_sha256": sha(flagged_file),
                                  "split_sha256": sha(splits / f"{name}.json")}


def rigid_fit(source: np.ndarray, target: np.ndarray, determinant: int):
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1., 1., determinant * np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt  # row-vector convention: target = source @ R + t
    translation = target.mean(0) - source.mean(0) @ rotation
    return rotation, translation


def choose_registration(commands: np.ndarray, xyz: np.ndarray, train: np.ndarray,
                        validation: np.ndarray, *, pcc_forward=pcc_xyz):
    """Compare translation, proper rigid and reflected rigid fits, including tendon orders.

    ``train`` and ``validation`` are positional indices. Test positions are never indexed.
    Identity order wins numerical ties, because the cable mapping is not identifiable
    under a free orthogonal mapping for a symmetric three-tendon PCC.
    """
    candidates = []
    identity = np.eye(3)
    for order in itertools.permutations(range(3)):
        local = pcc_forward(commands[:, order] / 10)
        for method in ("translation", "proper_rigid", "reflected_rigid"):
            if method == "translation":
                rotation = identity.copy()
                translation = (xyz[train] - local[train]).mean(0)
            else:
                rotation, translation = rigid_fit(local[train], xyz[train], 1 if method == "proper_rigid" else -1)
            error = local[validation] @ rotation + translation - xyz[validation]
            candidates.append({"method": method, "order": list(order),
                               "validation_RMSE_mm": error_metrics(error)["RMSE_norm"],
                               "row_rotation": rotation.tolist(), "translation_mm": translation.tolist()})
    priority = {"translation": 0, "proper_rigid": 1, "reflected_rigid": 2}
    best_score = min(c["validation_RMSE_mm"] for c in candidates)
    # Floating-point ties under cable symmetry are not physical evidence.
    tied = [c for c in candidates if c["validation_RMSE_mm"] <= best_score + 1e-8]
    chosen = min(tied, key=lambda c: (priority[c["method"]], c["order"]))
    local = pcc_forward(commands[:, chosen["order"]] / 10)
    prediction = local @ np.asarray(chosen["row_rotation"]) + np.asarray(chosen["translation_mm"])
    return prediction, chosen, candidates


def fit_nn(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray,
           y_scaler: StandardScaler, target_validation_mm: np.ndarray, seed: int,
           max_epochs: int, architectures=((64, 64), (215, 68))):
    """Choose architecture and epoch by validation RMSE in physical mm."""
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    xvt = torch.as_tensor(xv, dtype=torch.float32)
    trials = []
    for widths in architectures:
        torch.manual_seed(seed)
        net = FeedForwardNN(3, 3, tuple(widths), "relu")
        opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
        generator = torch.Generator().manual_seed(seed)
        history, best, stale, state, epoch_best = [], float("inf"), 0, None, 0
        for epoch in range(max_epochs):
            net.train()
            for batch in torch.randperm(len(xt), generator=generator).split(256):
                opt.zero_grad()
                loss = ((net(xt[batch]) - yt[batch]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                train_mse = float(((net(xt) - yt) ** 2).mean())
                prediction_scaled = net(xvt).numpy()
            val_mse = float(np.mean((prediction_scaled - yv) ** 2))
            val_rmse_mm = error_metrics(y_scaler.inverse_transform(prediction_scaled) - target_validation_mm)["RMSE_norm"]
            history.append({"epoch": epoch + 1, "train_MSE_scaled": train_mse,
                            "validation_MSE_scaled": val_mse, "validation_RMSE_mm": val_rmse_mm})
            if val_rmse_mm < best - 1e-6:
                best, stale, epoch_best = val_rmse_mm, 0, epoch + 1
                state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            else:
                stale += 1
            if stale >= 30:
                break
        net.load_state_dict(state)
        with torch.no_grad():
            val_scaled = net(xvt).numpy()
        trial = {"architecture": list(widths), "best_epoch": epoch_best,
                 "validation_RMSE_mm": best, "history": history,
                 "validation_prediction_scaled": val_scaled, "state": state}
        trials.append(trial)
    return trials


def run_one(name: str, data: Path, splits: Path, output: Path, models: Path,
            seed: int, epochs: int) -> None:
    result_dir, model_dir = output / name, models / name
    if result_dir.exists() or model_dir.exists():
        raise FileExistsError(f"{name} already has v2 outputs. Use a fresh --output and --models path for a new run")
    frame, split, (train_ids, val_ids, test_ids), hashes = load_inputs(data, splits, name)
    if split["seed"] != seed:
        raise ValueError(f"Saved {name} split uses seed {split['seed']}, requested {seed}")
    # Registration uses only train/validation XYZ. QA flags are descriptive, never fit selectors.
    commands = frame[legacy.INPUTS].to_numpy(float)
    xyz = frame[legacy.XYZ].to_numpy(float)
    train = frame.index.get_indexer(train_ids)
    val = frame.index.get_indexer(val_ids)
    test = frame.index.get_indexer(test_ids)
    registered, chosen, candidates = choose_registration(commands, xyz, train, val)
    residual = xyz - registered
    x = commands / 10
    x_scaler = StandardScaler().fit(x[train])
    y_scaler = StandardScaler().fit(residual[train])
    xs, xv, xe = (x_scaler.transform(x[idx]) for idx in (train, val, test))
    ys, yv = y_scaler.transform(residual[train]), y_scaler.transform(residual[val])
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    joblib.dump(x_scaler, model_dir / "input_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "residual_scaler.joblib")
    fitted = {"Linear": LinearRegression().fit(xs, ys)}
    selection = {}
    def score(model):
        return error_metrics(y_scaler.inverse_transform(model.predict(xv)) - residual[val])["RMSE_norm"]
    for degree in (2, 3):
        trials = []
        for alpha in (1e-4, 1e-2, 1., 100.):
            model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
            model.fit(xs, ys)
            trials.append((score(model), alpha, model))
        value, alpha, model = min(trials, key=lambda t: (t[0], t[1]))
        fitted[f"Ridge{degree}"] = model
        selection[f"Ridge{degree}"] = {"alpha": alpha, "validation_RMSE_mm": value}
    trials = []
    for k in (3, 7, 15, 31):
        model = KNeighborsRegressor(n_neighbors=k, weights="distance", algorithm="kd_tree").fit(xs, ys)
        trials.append((score(model), k, model))
    value, k, model = min(trials, key=lambda t: (t[0], t[1]))
    fitted["KNN"] = model
    selection["KNN"] = {"neighbors": k, "weights": "distance", "validation_RMSE_mm": value}
    for method, model in fitted.items():
        joblib.dump(model, model_dir / f"{method}.joblib")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    nn_trials = fit_nn(xs, ys, xv, yv, y_scaler, residual[val], seed, epochs)
    selected_nn = min(nn_trials, key=lambda t: (t["validation_RMSE_mm"], len(t["architecture"]), t["architecture"]))
    nn = FeedForwardNN(3, 3, tuple(selected_nn["architecture"]), "relu")
    nn.load_state_dict(selected_nn["state"])
    nn.eval()
    torch.save(selected_nn["state"], model_dir / "FFNN.pt")
    pd.DataFrame(selected_nn["history"]).to_csv(model_dir / "FFNN_history.csv", index=False)
    selection["FFNN"] = {"architecture": selected_nn["architecture"], "best_epoch": selected_nn["best_epoch"],
                         "validation_RMSE_mm": selected_nn["validation_RMSE_mm"], "candidate_scores": [
                             {"architecture": t["architecture"], "best_epoch": t["best_epoch"],
                              "validation_RMSE_mm": t["validation_RMSE_mm"]} for t in nn_trials]}
    with torch.no_grad():
        nn_residual = y_scaler.inverse_transform(nn(torch.as_tensor(xe, dtype=torch.float32)).numpy())
    corrections = {"PCC": np.zeros_like(residual[test]),
                   **{method: y_scaler.inverse_transform(model.predict(xe)) for method, model in fitted.items()},
                   "FFNN": nn_residual}
    predictions = frame.iloc[test][["row_id"] + legacy.INPUTS + legacy.XYZ + ["is_outlier"]].copy()
    metrics = []
    for method in METHODS:
        estimated = registered[test] + corrections[method]
        error = estimated - xyz[test]
        predictions[f"e_{method}"] = np.linalg.norm(error, axis=1)
        for axis, coordinate in enumerate("xyz"):
            predictions[f"{coordinate}_{method}"] = estimated[:, axis]
            predictions[f"d{coordinate}_{method}"] = corrections[method][:, axis]
        metrics.append({"experiment": name, "method": method, **error_metrics(error)})
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(result_dir / "metrics.csv", index=False)
    metric_frame.to_latex(result_dir / "metrics.tex", index=False, float_format="%.6f")
    plane_rows = []
    for zero in range(3):
        mask = predictions[legacy.INPUTS[zero]].to_numpy() == 0
        if not mask.any():
            continue
        for method in METHODS:
            predicted = predictions.loc[mask, [f"{axis}_{method}" for axis in "xyz"]].to_numpy(float)
            measured = predictions.loc[mask, legacy.XYZ].to_numpy(float)
            signed = predicted - measured
            plane_rows.append({"zero_tendon": zero + 1, "method": method,
                               "mean_signed_X_mm": float(signed[:, 0].mean()),
                               "mean_signed_Y_mm": float(signed[:, 1].mean()),
                               "mean_signed_Z_mm": float(signed[:, 2].mean()),
                               **error_metrics(signed)})
    pd.DataFrame(plane_rows).to_csv(result_dir / "plane_metrics.csv", index=False)
    save_json(result_dir / "registration.json", {"selected": chosen, "candidates": candidates,
        "fit_ids": train_ids.tolist(), "selection_ids": val_ids.tolist(), "test_ids": test_ids.tolist(),
        "assumptions": {"source_XYZ_unit": "mm (unverified)", "scale_fitted": False,
                        "PCC_geometry_mm": {"L": pcc_model.L, "d": pcc_model.d}, "marker": "PCC tip assumed"},
        "interpretation": "Train-registered PCC; cable order is an internal convention, not physical identification"})
    pd.DataFrame([{k: candidate[k] for k in ("method", "order", "validation_RMSE_mm")}
                  for candidate in candidates]).to_csv(result_dir / "registration_candidates.csv", index=False)
    save_json(model_dir / "metadata.json", {"version": VERSION, "experiment": name, "seed": seed,
        "epochs_max": epochs, "hashes": hashes, "registration_sha256": sha(result_dir / "registration.json"),
        "split": split, "selection": selection, "fit_ids": train_ids.tolist(), "validation_ids": val_ids.tolist(),
        "test_ids": test_ids.tolist(), "models": list(METHODS), "preprocessing": "input and residual scalers fit on training rows only",
        "QA_flags_used_for_training": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "torch": torch.__version__}})
    model_hashes = {path.name: sha(path) for path in sorted(model_dir.iterdir()) if path.is_file()}
    save_json(result_dir / "provenance.json", {"version": VERSION, "model_metadata_sha256": sha(model_dir / "metadata.json"),
        "model_file_sha256": model_hashes,
        "predictions_sha256": sha(result_dir / "predictions.csv"), "metrics_sha256": sha(result_dir / "metrics.csv"),
        "source_sha256": hashes["source_sha256"], "split_sha256": hashes["split_sha256"],
        "test_ids": test_ids.tolist(), "metrics_use": "full physical test set, including descriptive QA flags"})
    plot_one(name, predictions, metric_frame, result_dir, model_dir)


def plot_one(name: str, frame: pd.DataFrame, metrics: pd.DataFrame, output: Path, model_dir: Path) -> None:
    apply_thesis_style()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(frame.x_real, frame.y_real, frame.z_real, c=frame.e_PCC,
                         s=2, cmap="viridis")
    equal_3d_axes(ax, frame[legacy.XYZ].to_numpy(float))
    ax.set(xlabel="Measured X [assumed mm]", ylabel="Measured Y [assumed mm]",
           zlabel="Measured Z [assumed mm]", title=f"{name}: physical test positions")
    fig.colorbar(scatter, ax=ax, shrink=.65, label="Train-registered PCC error [mm]")
    save_figure(fig, output / "measured_workspace_3d.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in METHODS:
        e = frame[f"e_{method}"].to_numpy()
        axes[1].plot(np.sort(e), np.arange(1, len(e) + 1) / len(e), label=method)
        if method in ("PCC", "FFNN"):
            axes[0].hist(e, bins=50, alpha=.45, label=method)
    axes[0].set(xlabel="Position error [mm]", ylabel="Count", title=name)
    axes[1].set(xlabel="Position error [mm]", ylabel="Empirical CDF")
    for ax in axes:
        ax.legend(fontsize=8); light_grid(ax)
    save_figure(fig, output / "error_distribution.png")
    fig, axes = plt.subplots(3, len(METHODS), figsize=(20, 10), constrained_layout=True)
    vmax = max(frame[f"e_{method}"].max() for method in METHODS)
    for zero in range(3):
        active = [i for i in range(3) if i != zero]
        subset = frame[frame[legacy.INPUTS[zero]] == 0]
        for j, method in enumerate(METHODS):
            ax = axes[zero, j]
            sc = ax.scatter(subset[legacy.INPUTS[active[0]]] / 10, subset[legacy.INPUTS[active[1]]] / 10,
                            c=subset[f"e_{method}"], vmin=0, vmax=vmax, cmap="viridis", s=5)
            ax.set(xlim=(0, 10), ylim=(0, 10), xlabel=f"Tendon {active[0]+1} [mm]",
                   ylabel=f"Tendon {active[1]+1} [mm]", title=method)
            ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Held-out position error [mm]")
    save_figure(fig, output / "plane_error_maps.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, ax in zip("xyz", axes):
        measured = frame[f"{axis}_real"]
        for method in ("PCC", "Ridge3", "KNN", "FFNN"):
            ax.scatter(measured, frame[f"{axis}_{method}"], s=3, alpha=.25, label=method)
        low = min(measured.min(), *(frame[f"{axis}_{m}"].min() for m in ("PCC", "Ridge3", "KNN", "FFNN")))
        high = max(measured.max(), *(frame[f"{axis}_{m}"].max() for m in ("PCC", "Ridge3", "KNN", "FFNN")))
        ax.plot([low, high], [low, high], "k--", lw=1)
        ax.set(xlim=(low, high), ylim=(low, high), xlabel=f"Measured {axis.upper()} [mm]",
               ylabel=f"Predicted {axis.upper()} [mm]")
        light_grid(ax)
    axes[0].legend(fontsize=7)
    save_figure(fig, output / "coordinate_comparison.png")
    history = pd.read_csv(model_dir / "FFNN_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_MSE_scaled, label="Train")
    ax.plot(history.epoch, history.validation_MSE_scaled, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Scaled residual MSE", yscale="log")
    ax.legend(); light_grid(ax)
    save_figure(fig, output / "nn_training.png")
    if name.startswith("region"):
        zero = int(name[-1]) - 1
        active = [i for i in range(3) if i != zero]
        subset = frame[frame[legacy.INPUTS[zero]] == 0]
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        sc = ax.scatter(subset[legacy.INPUTS[active[0]]] / 10, subset[legacy.INPUTS[active[1]]] / 10,
                        c=subset.e_FFNN, vmin=0, vmax=vmax, cmap="viridis", s=9)
        ax.set(xlim=(0, 10), ylim=(0, 10), xlabel=f"Tendon {active[0]+1} [mm]",
               ylabel=f"Tendon {active[1]+1} [mm]", title="Excluded region: FFNN test error")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="Error [mm]")
        save_figure(fig, output / "held_out_region.png")


def report(data: Path, splits: Path, output: Path, models: Path) -> pd.DataFrame:
    """Aggregate completed v2 experiments; never trains or regenerates data."""
    tables = []
    for name in EXPERIMENTS:
        directory = output / name
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        metadata_file = models / name / "metadata.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if provenance["version"] != VERSION or metadata["version"] != VERSION:
            raise ValueError(f"Protocol version mismatch: {name}")
        for key, path in (("source_sha256", data / "Dataset-Actions-Positions.txt"),
                          ("split_sha256", splits / f"{name}.json"),
                          ("predictions_sha256", directory / "predictions.csv"),
                          ("metrics_sha256", directory / "metrics.csv"),
                          ("model_metadata_sha256", metadata_file)):
            if provenance[key] != sha(path):
                raise ValueError(f"Stale {name} artifact: {path}")
        if provenance["test_ids"] != metadata["test_ids"]:
            raise ValueError(f"Test ID mismatch: {name}")
        for filename, expected in provenance["model_file_sha256"].items():
            if sha(models / name / filename) != expected:
                raise ValueError(f"Stale {name} model artifact: {filename}")
        for filename in ("registration.json", "registration_candidates.csv", "plane_metrics.csv",
                         "error_distribution.png", "plane_error_maps.png", "coordinate_comparison.png",
                         "measured_workspace_3d.png", "nn_training.png"):
            if not (directory / filename).exists():
                raise ValueError(f"Missing {name} result: {filename}")
        if name.startswith("region") and not (directory / "held_out_region.png").exists():
            raise ValueError(f"Missing {name} held-out region figure")
        table = pd.read_csv(directory / "metrics.csv")
        prediction = pd.read_csv(directory / "predictions.csv")
        if prediction.row_id.tolist() != provenance["test_ids"]:
            raise ValueError(f"Prediction row order differs from test IDs: {name}")
        for method in METHODS:
            recorded = table[table.method == method]
            if len(recorded) != 1:
                raise ValueError(f"Missing or duplicate metric row: {name}/{method}")
            observed = error_metrics(prediction[[f"{axis}_{method}" for axis in "xyz"]].to_numpy(float)
                                     - prediction[legacy.XYZ].to_numpy(float))
            if not np.isclose(recorded.iloc[0].RMSE_norm, observed["RMSE_norm"], atol=1e-8, rtol=0):
                raise ValueError(f"Metric/prediction mismatch: {name}/{method}")
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "all_metrics.csv", index=False)
    combined.to_latex(output / "all_metrics.tex", index=False, float_format="%.6f")
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        subset = combined[(combined.method == method) & combined.experiment.str.startswith("sparse_")].copy()
        subset["step_mm"] = subset.experiment.str.split("_").str[-1].astype(int) / 10
        subset = subset.sort_values("step_mm")
        ax.plot(subset.step_mm, subset.RMSE_norm, marker="o", label=method)
    ax.set(xlabel="Training grid spacing [mm]", ylabel="Test RMSE [mm]")
    ax.legend(); light_grid(ax)
    save_figure(fig, output / "sparse_grid_accuracy.png")
    primary = combined[combined.experiment == "random"].set_index("method")
    lines = ["# Phase 6 review v2: physical test results", "",
             "These results use original physical measurements, an assumed mm XYZ unit, and a PCC transform fitted on training measurements and selected on validation measurements. They do not identify the physical cable numbering or independently calibrate the sensor frame.", "",
             f"RQ1: Train-registered PCC RMSE = {primary.loc['PCC', 'RMSE_norm']:.6f} mm.",
             f"RQ2: FFNN RMSE = {primary.loc['FFNN', 'RMSE_norm']:.6f} mm; compare against PCC on the same {int(primary.loc['PCC', 'N'])} test rows.",
             f"RQ3: Linear {primary.loc['Linear', 'RMSE_norm']:.6f}, Ridge2 {primary.loc['Ridge2', 'RMSE_norm']:.6f}, Ridge3 {primary.loc['Ridge3', 'RMSE_norm']:.6f}, KNN {primary.loc['KNN', 'RMSE_norm']:.6f}, FFNN {primary.loc['FFNN', 'RMSE_norm']:.6f} mm RMSE. A single static dataset and one split do not establish statistical significance.",
             "RQ4: Sparse-grid test RMSE [mm]:", "",
             combined[combined.experiment.str.startswith("sparse_")][["experiment", "method", "N", "RMSE_norm"]].to_string(index=False),
             "", "RQ5: Excluded-region test RMSE [mm]:", "",
             combined[combined.experiment.str.startswith("region_")][["experiment", "method", "N", "RMSE_norm"]].to_string(index=False),
             "", "Limitations: no external coordinate/marker calibration, unknown measurement repeatability and history, one fixed split seed, no three-active-tendon observations, and no physical inverse-kinematics validation."]
    (output / "research_answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined


def compare_old_new(data: Path, splits: Path, output: Path, models: Path,
                    old: Path = ROOT / "results" / "final" / "phase5_real_compensation") -> pd.DataFrame:
    """Paired comparison on identical physical test rows; reject stale old outputs."""
    report(data, splits, output, models)
    rows = []
    for name in EXPERIMENTS:
        old_predictions = pd.read_csv(old / name / "predictions_all.csv")
        new_predictions = pd.read_csv(output / name / "predictions.csv")
        old_provenance = json.loads((data / "experiments" / name / "prediction_provenance.json").read_text())
        if old_provenance["input_sha256"] != sha(data / "real_dataset_flagged.csv"):
            raise ValueError(f"Old {name} inputs do not match current source processing")
        if old_provenance["prediction_sha256"] != sha(data / "experiments" / name / "real_vs_pcc.csv"):
            raise ValueError(f"Old {name} registered positions changed")
        if set(old_predictions.row_id) != set(new_predictions.row_id):
            raise ValueError(f"Old/new test IDs differ: {name}")
        old_predictions = old_predictions.set_index("row_id").loc[new_predictions.row_id]
        if not np.allclose(old_predictions[legacy.XYZ].to_numpy(), new_predictions[legacy.XYZ].to_numpy(), rtol=0, atol=1e-8):
            raise ValueError(f"Old/new measured XYZ differ: {name}")
        for method in ("PCC", "Linear", "Ridge2", "Ridge3", "FFNN"):
            old_error = old_predictions[f"e_{method}"].to_numpy()
            new_error = new_predictions[f"e_{method}"].to_numpy()
            rows.append({"experiment": name, "method": method, "N": len(new_error),
                         "old_RMSE_mm": float(np.sqrt(np.mean(old_error ** 2))),
                         "new_RMSE_mm": float(np.sqrt(np.mean(new_error ** 2))),
                         "old_MAE_mm": float(np.mean(old_error)), "new_MAE_mm": float(np.mean(new_error)),
                         "mean_paired_error_change_mm": float(np.mean(new_error - old_error))})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "paired_legacy_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run", "report", "compare"))
    parser.add_argument("--experiment", choices=("all",) + EXPERIMENTS, default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    if args.stage == "check":
        for name in names:
            frame, split, groups, hashes = load_inputs(args.data, args.splits, name)
            print(f"{name}: train={len(groups[0])}, validation={len(groups[1])}, test={len(groups[2])}, source={hashes['source_sha256']}")
    elif args.stage == "run":
        if any((args.output / name).exists() or (args.models / name).exists() for name in names):
            parser.error("A requested v2 experiment already exists; choose fresh --output and --models paths")
        for name in names:
            run_one(name, args.data, args.splits, args.output, args.models, args.seed, args.epochs)
            print(f"completed {name}", flush=True)
    elif args.stage == "report":
        report(args.data, args.splits, args.output, args.models)
    else:
        compare_old_new(args.data, args.splits, args.output, args.models)


if __name__ == "__main__":
    main()
