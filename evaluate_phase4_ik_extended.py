"""Final synthetic-reference IK validation for PCC and PCC+NN control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase4_inverse_kinematics import (
    CANONICAL_MODEL_PATH,
    CANONICAL_X_SCALER_PATH,
    CANONICAL_Y_SCALER_PATH,
    IKConfig,
    Phase3NNCorrector,
    REAL_MODEL_PARAMS,
    inverse_kinematics,
    make_pcc_nn_forward,
    pcc_forward_xyz,
    real_forward_xyz,
    sample_valid_dls,
)
from synthetic_validation_common import error_metrics, save_metric_table


DEFAULT_OUT = Path("outputs/phase4_ik_validation")
METHODS = ("PCC IK", "PCC+NN IK")


def target_commands(n: int, seed: int) -> tuple[np.ndarray, list[str]]:
    if n < 50:
        raise ValueError("At least 50 targets are required")
    n_boundary = max(15, int(np.ceil(n * 0.30)))
    ordinary = sample_valid_dls(n - n_boundary, seed)
    rng = np.random.default_rng(seed + 1)
    boundary = np.zeros((n_boundary, 3), dtype=np.float64)
    pairs = ((0, 1), (0, 2), (1, 2))
    for i in range(n_boundary):
        a, b = pairs[i % len(pairs)]
        boundary[i, a] = rng.uniform(9.0, 10.0)
        boundary[i, b] = rng.uniform(7.0, 10.0)
    commands = np.vstack([ordinary, boundary])
    labels = ["random"] * len(ordinary) + ["boundary_large_bend"] * n_boundary
    return commands, labels


def make_plots(results: pd.DataFrame, out_dir: Path) -> None:
    colors = {"PCC IK": "#bd6546", "PCC+NN IK": "#257a77"}
    first = results[results.method == METHODS[0]].sort_values("target_index")
    target = first[["target_x", "target_y", "target_z"]].to_numpy()
    all_positions = np.vstack([target, results[["reached_x", "reached_y", "reached_z"]].to_numpy()])
    limits = [(float(all_positions[:, i].min()), float(all_positions[:, i].max())) for i in range(3)]

    for method, filename in [
        ("PCC IK", "pcc_ik_target_vs_reached_3d.png"),
        ("PCC+NN IK", "pcc_nn_ik_target_vs_reached_3d.png"),
    ]:
        sub = results[results.method == method].sort_values("target_index")
        reached = sub[["reached_x", "reached_y", "reached_z"]].to_numpy()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*target.T, color="black", s=19, label="Target")
        ax.scatter(*reached.T, color=colors[method], s=14, alpha=0.75, label=method)
        ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, float(results.err_norm.max()), 26)
    for method in METHODS:
        sub = results[results.method == method]
        ax.hist(sub.err_norm, bins=bins, alpha=0.6, color=colors[method], label=method)
    ax.set(xlabel="IK error norm [mm]", ylabel="Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ik_error_histogram.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([results[results.method == method].err_norm for method in METHODS],
               showfliers=False)
    ax.set_xticks([1, 2], list(METHODS))
    ax.set(ylabel="IK error norm [mm]")
    fig.tight_layout()
    fig.savefig(out_dir / "ik_error_boxplot.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    indices = first.target_index.to_numpy()
    for axis, panel in zip("xyz", axes):
        panel.plot(indices, first[f"target_{axis}"], color="black", linewidth=1.5,
                   label="Target")
        for method in METHODS:
            sub = results[results.method == method].sort_values("target_index")
            panel.plot(indices, sub[f"reached_{axis}"], color=colors[method],
                       linewidth=1, alpha=0.8, label=method)
        panel.set_ylabel(f"{axis.upper()} [mm]")
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Ordered target index (sample order)")
    fig.tight_layout()
    fig.savefig(out_dir / "target_vs_reached_trajectory.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        sub = results[results.method == method]
        ax.scatter(sub.ik_dl_norm, sub.err_norm, s=17, alpha=0.75,
                   color=colors[method], label=method)
    ax.set(xlabel="Norm of IK command [mm]", ylabel="IK error norm [mm]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_configuration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-targets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-noise-sigma", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.real_noise_sigma < 0:
        parser.error("--real-noise-sigma must be nonnegative")

    cfg = IKConfig(device=args.device)
    corrector = Phase3NNCorrector(CANONICAL_MODEL_PATH, CANONICAL_X_SCALER_PATH,
                                  CANONICAL_Y_SCALER_PATH, device=args.device)
    forward = {
        "PCC IK": pcc_forward_xyz,
        "PCC+NN IK": make_pcc_nn_forward(corrector),
    }
    commands, labels = target_commands(args.n_targets, args.seed)
    rows = []
    for i, (source_dl, group) in enumerate(zip(commands, labels)):
        target_seed = None if args.real_noise_sigma == 0 else args.seed + 10000 + i
        target = real_forward_xyz(source_dl, args.real_noise_sigma, target_seed)
        for method in METHODS:
            ik = inverse_kinematics(target, forward[method], cfg)
            # Common measurement noise for the two methods; independent of target noise.
            reached_seed = None if args.real_noise_sigma == 0 else args.seed + 20000 + i
            reached = real_forward_xyz(ik.dl, args.real_noise_sigma, reached_seed)
            err = reached - target
            rows.append({
                "target_index": i, "target_group": group, "method": method,
                **{f"source_dl{j+1}": float(source_dl[j]) for j in range(3)},
                **{f"target_{axis}": float(target[j]) for j, axis in enumerate("xyz")},
                **{f"ik_dl{j+1}": float(ik.dl[j]) for j in range(3)},
                "ik_dl_norm": float(np.linalg.norm(ik.dl)),
                "ik_residual_norm": float(ik.residual_norm),
                **{f"reached_{axis}": float(reached[j]) for j, axis in enumerate("xyz")},
                **{f"err_{axis}": float(err[j]) for j, axis in enumerate("xyz")},
                "err_norm": float(np.linalg.norm(err)), "ik_success": bool(ik.success),
                "solver": ik.solver,
            })

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "phase4_ik_results.csv", index=False)
    metrics = []
    boundary_metrics = []
    for method in METHODS:
        sub = results[results.method == method]
        metrics.append({"method": method, **error_metrics(sub[["err_x", "err_y", "err_z"]].to_numpy())})
        boundary = sub[sub.target_group == "boundary_large_bend"]
        boundary_metrics.append({"method": method, **error_metrics(boundary[["err_x", "err_y", "err_z"]].to_numpy())})
    table = save_metric_table(metrics, out_dir, "phase4_ik_metrics",
                              "Phase 4 IK target-to-reached error (mm).", "tab:phase4_extended")
    make_plots(results, out_dir)
    summary = {
        "reference_model": "synthetic reference model (perturbed PCC; no physical robot data)",
        "real_model_used_inside_ik": False,
        "methods_compared": list(METHODS),
        "n_targets": args.n_targets, "n_boundary_large_bend": labels.count("boundary_large_bend"),
        "seed": args.seed, "noise_sigma_mm": args.real_noise_sigma,
        "noise_protocol": "Independent target and reached Gaussian noise; shared reached-noise draw across methods for each target",
        "error_definition": "p_reached = f_syn(dl_IK); e_IK = p_reached - p_target",
        "trajectory_plot_order": "sample index; targets are sampled points, not a continuous planned path",
        "constraints": {"dl_min": cfg.dl_min, "dl_max": cfg.dl_max,
                        "max_active_tendons": cfg.max_active_tendons},
        "canonical_phase3_model": {"model_path": str(CANONICAL_MODEL_PATH),
                                   "x_scaler_path": str(CANONICAL_X_SCALER_PATH),
                                   "y_scaler_path": str(CANONICAL_Y_SCALER_PATH)},
        "real_model_parameters": {**REAL_MODEL_PARAMS, "sigma_noise": args.real_noise_sigma},
        "metrics": metrics, "boundary_metrics": boundary_metrics,
        "ik_solver_success_counts": {method: int(results[(results.method == method) & results.ik_success].shape[0])
                                     for method in METHODS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
