"""Independent synthetic studies of dataset size, noise, range, and NN architecture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase4_inverse_kinematics import (
    CANONICAL_MODEL_PATH,
    CANONICAL_X_SCALER_PATH,
    CANONICAL_Y_SCALER_PATH,
    Phase3NNCorrector,
    REAL_MODEL_PARAMS,
)
from synthetic_validation_common import (
    error_metrics,
    pcc_xyz,
    sample_commands,
    save_metric_table,
    synthetic_xyz,
)
from train_nn_phase3 import FeedForwardNN


FULL_SIZES = [500, 1000, 2000, 5000, 10000, 20000]
QUICK_SIZES = [500, 1000, 2000]
NOISE_SIGMAS = [0.0, 0.2, 0.5, 0.8, 1.0]
ARCHITECTURES = {"small": (32, 32), "medium": (128, 64),
                 "large": (256, 128, 64)}


def features(dls: np.ndarray) -> np.ndarray:
    return np.hstack([dls, (dls > 1e-9).astype(np.float64)])


def dataset(n: int, dl_min: float, dl_max: float, sigma: float,
            command_seed: int, noise_seed: int) -> tuple[np.ndarray, np.ndarray]:
    dls = sample_commands(n, command_seed, dl_min, dl_max)
    residual = synthetic_xyz(dls, sigma, noise_seed) - pcc_xyz(dls)
    return dls, residual


def fit_predict(X_pool: np.ndarray, Y_pool: np.ndarray, X_test: np.ndarray,
                hidden: tuple[int, ...], seed: int, epochs: int, patience: int,
                device: str) -> tuple[np.ndarray, dict]:
    """Fit only on a train/validation split of the supplied pool."""
    train_idx, val_idx = train_test_split(np.arange(len(X_pool)), test_size=0.15,
                                          random_state=seed, shuffle=True)
    x_scaler = StandardScaler().fit(features(X_pool[train_idx]))
    y_scaler = StandardScaler().fit(Y_pool[train_idx])
    x_train = torch.tensor(x_scaler.transform(features(X_pool[train_idx])), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(Y_pool[train_idx]), dtype=torch.float32)
    x_val = torch.tensor(x_scaler.transform(features(X_pool[val_idx])), dtype=torch.float32, device=device)
    y_val = torch.tensor(y_scaler.transform(Y_pool[val_idx]), dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
                        generator=generator)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FeedForwardNN(in_dim=6, hidden_sizes=hidden, activation="relu").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_scaler.transform(features(X_test)), dtype=torch.float32, device=device)
        pred = model(x_test).cpu().numpy()
    return y_scaler.inverse_transform(pred), {
        "train": int(len(train_idx)), "validation": int(len(val_idx)),
        "best_epoch": best_epoch, "best_validation_scaled_mse": best_loss,
    }


def line_plot(x, y, xlabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", color="#257a77", linewidth=1.8)
    ax.set(xlabel=xlabel, ylabel="RMSE of Cartesian error norm [mm]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def box_plot(errors: dict[str, np.ndarray], out: Path, xlabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([np.linalg.norm(e, axis=1) for e in errors.values()], showfliers=False)
    ax.set_xticks(np.arange(1, len(errors) + 1), list(errors))
    ax.set(xlabel=xlabel, ylabel="Cartesian error norm [mm]")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Short pilot run (default).")
    mode.add_argument("--full", action="store_true", help="All requested dataset sizes and longer training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/summer_experiments"))
    args = parser.parse_args()
    full = bool(args.full)
    sizes = FULL_SIZES if full else QUICK_SIZES
    epochs = args.epochs or (100 if full else 30)
    if epochs < 1:
        parser.error("--epochs must be positive")
    patience = 15 if full else 8
    n_test = 2000 if full else 500
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # Dataset-size study: nested fitting pools, one independent fixed test set.
    X_pool, Y_pool = dataset(max(sizes), 0, 10, 0.5, seed + 10, seed + 11)
    X_test, Y_test = dataset(n_test, 0, 10, 0.5, seed + 20, seed + 21)
    size_rows, size_training = [], {}
    for n in sizes:
        pred, details = fit_predict(X_pool[:n], Y_pool[:n], X_test, (128, 64),
                                    seed, epochs, patience, args.device)
        size_rows.append({"dataset_size": n, "method": "PCC+NN",
                          **error_metrics(pred - Y_test)})
        size_training[str(n)] = details
        print(f"Dataset size {n} complete")
    save_metric_table(size_rows, out, "dataset_size_metrics",
                      "Influence of fitting-pool size on synthetic test error (mm).",
                      "tab:summer_dataset_size")
    line_plot(sizes, [row["RMSE_norm"] for row in size_rows],
              "Fitting-pool size N", out / "dataset_size_rmse.png")

    # Noise study: evaluate the existing canonical correction model only.
    corrector = Phase3NNCorrector(CANONICAL_MODEL_PATH, CANONICAL_X_SCALER_PATH,
                                  CANONICAL_Y_SCALER_PATH, device=args.device)
    noise_commands = sample_commands(n_test, seed + 30)
    noise_pcc = pcc_xyz(noise_commands)
    noise_pred = corrector.predict_delta(noise_commands)
    noise_rows = []
    for sigma in NOISE_SIGMAS:
        y = synthetic_xyz(noise_commands, sigma, seed + 31) - noise_pcc
        noise_rows.append({"sigma_mm": sigma, "method": "PCC+NN",
                           "training_mode": "evaluated_only", **error_metrics(noise_pred - y)})
    save_metric_table(noise_rows, out, "noise_metrics",
                      "Canonical NN evaluated under synthetic measurement noise (mm).",
                      "tab:summer_noise")
    line_plot(NOISE_SIGMAS, [row["RMSE_norm"] for row in noise_rows],
              "Gaussian noise sigma [mm]", out / "noise_rmse.png")

    # Extrapolation study: new model trained on [0, 8], never on (8, 10].
    gen_train_n = 8000 if full else 2000
    gen_test_n = 1000 if full else 500
    X_gen_train, Y_gen_train = dataset(gen_train_n, 0, 8, 0, seed + 40, seed + 41)
    X_in, Y_in = dataset(gen_test_n, 0, 8, 0, seed + 42, seed + 43)
    X_out, Y_out = dataset(gen_test_n, 8, 10, 0, seed + 44, seed + 45)
    gen_pred, gen_details = fit_predict(X_gen_train, Y_gen_train,
                                        np.vstack([X_in, X_out]), (128, 64),
                                        seed, epochs, patience, args.device)
    gen_rows, gen_errors = [], {}
    for label, truth, predicted in [
        ("inside_0_8", Y_in, gen_pred[:gen_test_n]),
        ("outside_8_10", Y_out, gen_pred[gen_test_n:]),
    ]:
        for method, err in [("PCC", -truth), ("PCC+NN", predicted - truth)]:
            gen_rows.append({"test_range": label, "method": method,
                             **error_metrics(err)})
            gen_errors[f"{label} {method}"] = err
    save_metric_table(gen_rows, out, "generalization_metrics",
                      "Interpolation and extrapolation in tendon shortening (mm).",
                      "tab:summer_generalization")
    box_plot(gen_errors, out / "generalization_boxplot.png")

    # Architecture study: identical fitting pool, validation split, and test set.
    arch_n = 5000 if full else 2000
    arch_rows, arch_errors, arch_training = [], {}, {}
    for label, hidden in ARCHITECTURES.items():
        pred, details = fit_predict(X_pool[:arch_n], Y_pool[:arch_n], X_test,
                                    hidden, seed, epochs, patience, args.device)
        err = pred - Y_test
        arch_rows.append({"architecture": label, "hidden_layers": str(list(hidden)),
                          **error_metrics(err)})
        arch_errors[label] = err
        arch_training[label] = details
        print(f"Architecture {label} complete")
    save_metric_table(arch_rows, out, "architecture_metrics",
                      "Feedforward NN architecture comparison (mm).",
                      "tab:summer_architecture")
    box_plot(arch_errors, out / "architecture_boxplot.png")

    summary = {
        "reference_model": "synthetic reference model (perturbed PCC; no physical robot data)",
        "mode": "full" if full else "quick",
        "seed": seed, "device": args.device, "epochs_max": epochs,
        "early_stopping_patience": patience, "held_out_test_size": n_test,
        "input_features": "dl1, dl2, dl3 plus three activity indicators",
        "target": "synthetic reference position minus PCC position",
        "data_generation": "independent seeded synthetic samples with max two active tendons; same perturbation parameters as canonical Phase 2",
        "normalization": "input and residual scalers fit on training split only",
        "train_validation_fraction": 0.15,
        "dataset_size": {"mode": "retrained", "noise_sigma_mm": 0.5,
                         "fitting_pool_sizes": sizes,
                         "test_set_shared_across_sizes": True,
                         "training_details": size_training, "metrics": size_rows},
        "noise": {"mode": "evaluated_only", "retrained": False,
                  "canonical_model": str(CANONICAL_MODEL_PATH),
                  "sigmas_mm": NOISE_SIGMAS, "metrics": noise_rows},
        "generalization": {"mode": "retrained", "train_range_mm": [0, 8],
                           "outside_test_range_mm": "active tendon shortenings in (8, 10]",
                           "noise_sigma_mm": 0, "training_details": gen_details,
                           "metrics": gen_rows},
        "architecture": {"mode": "retrained", "same_split_for_all": True,
                         "fitting_pool_size": arch_n, "training_details": arch_training,
                         "metrics": arch_rows},
        "synthetic_reference_parameters": REAL_MODEL_PARAMS,
        "limitations": "Synthetic-only results; quick mode is a pilot and does not cover all requested dataset sizes.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
