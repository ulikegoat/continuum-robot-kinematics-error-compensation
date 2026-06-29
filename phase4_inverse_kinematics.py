from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import pcc_model as pcc

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - exercised only when scipy is absent
    least_squares = None


ForwardFn = Callable[[np.ndarray], np.ndarray]

CANONICAL_MODEL_DIR = Path("phase3_last_16_03_2026")
CANONICAL_MODEL_PATH = CANONICAL_MODEL_DIR / "nn_model.pt"
CANONICAL_X_SCALER_PATH = CANONICAL_MODEL_DIR / "x_scaler.pkl"
CANONICAL_Y_SCALER_PATH = CANONICAL_MODEL_DIR / "y_scaler.pkl"
RESULTS_DIR = Path("phase4_results")

DL_MIN = 0.0
DL_MAX = 10.0
EPS_ACTIVITY = 1e-9

REAL_MODEL_PARAMS = {
    "alpha_per_m": 3.5,
    "beta_rad_per_m": 3.5,
    "offset": [0.5, 0.5, 0.3],
    "sigma_noise": 0.5,
    "theta_max_deg": 95.0,
}


def get_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
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


@dataclass
class IKConfig:
    model_path: Path = CANONICAL_MODEL_PATH
    x_scaler_path: Path = CANONICAL_X_SCALER_PATH
    y_scaler_path: Path = CANONICAL_Y_SCALER_PATH
    dl_min: float = DL_MIN
    dl_max: float = DL_MAX
    max_active_tendons: int = 2
    grid_n: int = 9
    n_starts_per_mode: int = 3
    max_nfev: int = 120
    dls_max_iter: int = 80
    jac_eps: float = 1e-4
    damping: float = 1e-3
    eps_activity: float = EPS_ACTIVITY
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IKResult:
    target_xyz: np.ndarray
    dl: np.ndarray
    forward_xyz: np.ndarray
    residual_xyz: np.ndarray
    residual_norm: float
    active_tendons: tuple[int, ...]
    active_set: tuple[int, ...]
    solver: str
    nfev: int
    success: bool

    def to_dict(self) -> dict:
        return {
            "target_xyz": self.target_xyz.tolist(),
            "dl": self.dl.tolist(),
            "forward_xyz": self.forward_xyz.tolist(),
            "residual_xyz": self.residual_xyz.tolist(),
            "residual_norm": self.residual_norm,
            "active_tendons": [int(i + 1) for i in self.active_tendons],
            "active_set": [int(i + 1) for i in self.active_set],
            "solver": self.solver,
            "nfev": self.nfev,
            "success": self.success,
        }


class Phase3NNCorrector:
    def __init__(
        self,
        model_path: Path = CANONICAL_MODEL_PATH,
        x_scaler_path: Path = CANONICAL_X_SCALER_PATH,
        y_scaler_path: Path = CANONICAL_Y_SCALER_PATH,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.x_scaler_path = x_scaler_path
        self.y_scaler_path = y_scaler_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._check_artifacts()

        self.x_scaler = joblib.load(self.x_scaler_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", {})
        self.use_activity_mask = bool(model_cfg.get("use_activity_mask", True))
        self.eps_activity = float(model_cfg.get("eps_activity", EPS_ACTIVITY))
        self.in_dim = int(model_cfg.get("in_dim", 6 if self.use_activity_mask else 3))
        hidden_sizes = tuple(model_cfg.get("hidden_sizes", (64, 64)))
        activation = str(model_cfg.get("activation", "relu"))

        self.model = FeedForwardNN(
            in_dim=self.in_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _check_artifacts(self) -> None:
        for path in [self.model_path, self.x_scaler_path, self.y_scaler_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing canonical Phase 3 artifact: {path.resolve()}")

    @torch.no_grad()
    def predict_delta(self, dls: np.ndarray) -> np.ndarray:
        dls = np.asarray(dls, dtype=np.float64)
        if dls.ndim == 1:
            dls = dls.reshape(1, 3)
        if dls.ndim != 2 or dls.shape[1] != 3:
            raise ValueError(f"dls must have shape (N,3), got {dls.shape}")

        feats = dls
        if self.use_activity_mask:
            activity = (np.abs(dls) > self.eps_activity).astype(np.float64)
            feats = np.hstack([dls, activity])

        if feats.shape[1] != self.in_dim:
            raise ValueError(f"Feature dimension mismatch: {feats.shape[1]} vs {self.in_dim}")

        feats_n = self.x_scaler.transform(feats).astype(np.float32)
        pred_n = self.model(torch.from_numpy(feats_n).to(self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(pred_n).astype(np.float64)


def pcc_forward_xyz(dl: Iterable[float]) -> np.ndarray:
    dl = np.asarray(list(dl), dtype=np.float64)
    xyz_theta = pcc.pcc_forward(float(dl[0]), float(dl[1]), float(dl[2]))
    return np.array(xyz_theta[:3], dtype=np.float64)


def make_pcc_nn_forward(corrector: Phase3NNCorrector) -> ForwardFn:
    def forward(dl: np.ndarray) -> np.ndarray:
        dl = np.asarray(dl, dtype=np.float64)
        return pcc_forward_xyz(dl) + corrector.predict_delta(dl.reshape(1, 3))[0]

    return forward


def active_sets(max_active: int = 2) -> list[tuple[int, ...]]:
    if max_active != 2:
        raise ValueError("This project enforces max 2 active tendons.")
    return [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


def dl_from_vars(q: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    dl = np.zeros(3, dtype=np.float64)
    if active:
        dl[list(active)] = q
    return dl


def vars_from_dl(dl: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    if not active:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(dl, dtype=np.float64)[list(active)]


def coarse_grid_for_active_set(active: tuple[int, ...], cfg: IKConfig) -> np.ndarray:
    if not active:
        return np.zeros((1, 3), dtype=np.float64)

    values = np.linspace(cfg.dl_min, cfg.dl_max, cfg.grid_n)
    if len(active) == 1:
        rows = np.zeros((cfg.grid_n, 3), dtype=np.float64)
        rows[:, active[0]] = values
        return rows

    rows = []
    for a in values:
        for b in values:
            dl = np.zeros(3, dtype=np.float64)
            dl[active[0]] = a
            dl[active[1]] = b
            rows.append(dl)
    return np.vstack(rows)


def best_grid_seeds(
    target: np.ndarray,
    forward_fn: ForwardFn,
    active: tuple[int, ...],
    cfg: IKConfig,
) -> np.ndarray:
    grid = coarse_grid_for_active_set(active, cfg)
    xyz = np.vstack([forward_fn(dl) for dl in grid])
    errs = np.linalg.norm(xyz - target.reshape(1, 3), axis=1)
    n = min(cfg.n_starts_per_mode, len(grid))
    idx = np.argpartition(errs, n - 1)[:n]
    idx = idx[np.argsort(errs[idx])]
    return grid[idx]


def numerical_jacobian(
    q: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    cfg: IKConfig,
) -> np.ndarray:
    r0 = residual_fn(q)
    jac = np.zeros((r0.size, q.size), dtype=np.float64)
    for j in range(q.size):
        step = cfg.jac_eps * max(1.0, abs(float(q[j])))
        qp = q.copy()
        qm = q.copy()
        qp[j] = min(cfg.dl_max, qp[j] + step)
        qm[j] = max(cfg.dl_min, qm[j] - step)
        denom = qp[j] - qm[j]
        if denom <= 0:
            continue
        jac[:, j] = (residual_fn(qp) - residual_fn(qm)) / denom
    return jac


def damped_least_squares(
    q0: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    cfg: IKConfig,
) -> tuple[np.ndarray, int, bool]:
    q = np.clip(np.asarray(q0, dtype=np.float64), cfg.dl_min, cfg.dl_max)
    nfev = 1
    r = residual_fn(q)
    best = float(np.linalg.norm(r))

    for _ in range(cfg.dls_max_iter):
        jac = numerical_jacobian(q, residual_fn, cfg)
        lhs = jac.T @ jac + cfg.damping * np.eye(q.size)
        rhs = -jac.T @ r
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            q_trial = np.clip(q + scale * step, cfg.dl_min, cfg.dl_max)
            r_trial = residual_fn(q_trial)
            nfev += 1
            err = float(np.linalg.norm(r_trial))
            if err < best:
                q, r, best = q_trial, r_trial, err
                accepted = True
                break

        if not accepted or np.linalg.norm(step) < 1e-7 or best < 1e-6:
            return q, nfev, best < 1e-4

    return q, nfev, best < 1e-4


def solve_active_set(
    target: np.ndarray,
    forward_fn: ForwardFn,
    active: tuple[int, ...],
    start_dl: np.ndarray,
    cfg: IKConfig,
) -> tuple[np.ndarray, int, bool, str]:
    if not active:
        return np.zeros(3, dtype=np.float64), 1, True, "constant"

    q0 = vars_from_dl(start_dl, active)

    def residual_q(q: np.ndarray) -> np.ndarray:
        return forward_fn(dl_from_vars(q, active)) - target

    if least_squares is not None:
        res = least_squares(
            residual_q,
            q0,
            bounds=(cfg.dl_min, cfg.dl_max),
            max_nfev=cfg.max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        return dl_from_vars(res.x, active), int(res.nfev), bool(res.success), "scipy.least_squares"

    q, nfev, success = damped_least_squares(q0, residual_q, cfg)
    return dl_from_vars(q, active), nfev, success, "damped_least_squares"


def inverse_kinematics(
    target: Iterable[float],
    forward_fn: ForwardFn,
    cfg: IKConfig | None = None,
) -> IKResult:
    cfg = cfg or IKConfig()
    target_arr = np.asarray(list(target), dtype=np.float64)
    if target_arr.shape != (3,):
        raise ValueError(f"target must have shape (3,), got {target_arr.shape}")

    best: IKResult | None = None

    for active in active_sets(cfg.max_active_tendons):
        for seed in best_grid_seeds(target_arr, forward_fn, active, cfg):
            dl, nfev, success, solver = solve_active_set(target_arr, forward_fn, active, seed, cfg)
            xyz = forward_fn(dl)
            residual = xyz - target_arr
            residual_norm = float(np.linalg.norm(residual))
            actual_active = tuple(np.where(np.abs(dl) > cfg.eps_activity)[0].tolist())
            result = IKResult(
                target_xyz=target_arr,
                dl=dl,
                forward_xyz=xyz,
                residual_xyz=residual,
                residual_norm=residual_norm,
                active_tendons=actual_active,
                active_set=active,
                solver=solver,
                nfev=nfev,
                success=success,
            )
            if best is None or result.residual_norm < best.residual_norm:
                best = result

    if best is None:
        raise RuntimeError("IK produced no candidate solution.")
    return best


def configure_real_model(noise_sigma: float):
    # real_model is used only by target generation and final command evaluation.
    import real_model as real

    real.alpha_per_m = REAL_MODEL_PARAMS["alpha_per_m"]
    real.beta_rad_per_m = REAL_MODEL_PARAMS["beta_rad_per_m"]
    real.offset = np.array(REAL_MODEL_PARAMS["offset"], dtype=float)
    real.sigma_noise = float(noise_sigma)
    real.theta_max = np.radians(REAL_MODEL_PARAMS["theta_max_deg"])
    return real


def real_forward_xyz(dl: Iterable[float], noise_sigma: float = 0.0, seed: int | None = None) -> np.ndarray:
    real = configure_real_model(noise_sigma)
    if seed is not None:
        np.random.seed(seed)
    dl = np.asarray(list(dl), dtype=np.float64)
    x, y, z, _theta = real.real_forward(float(dl[0]), float(dl[1]), float(dl[2]), enforce_limit=True)
    return np.array([x, y, z], dtype=np.float64)


def sample_valid_dls(n: int, seed: int, dl_min: float = DL_MIN, dl_max: float = DL_MAX) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    pair_modes = [(0, 1), (0, 2), (1, 2)]
    while len(rows) < n:
        dl = np.zeros(3, dtype=np.float64)
        if rng.random() < 0.4:
            active = (int(rng.integers(0, 3)),)
        else:
            active = pair_modes[int(rng.integers(0, len(pair_modes)))]
        for idx in active:
            dl[idx] = float(rng.uniform(dl_min, dl_max))
        rows.append(dl)
    return np.vstack(rows)


def generate_targets(n: int, seed: int, noise_sigma: float) -> pd.DataFrame:
    source_dls = sample_valid_dls(n, seed)
    rows = []
    for i, dl in enumerate(source_dls):
        real_seed = None if noise_sigma == 0 else seed + i
        target = real_forward_xyz(dl, noise_sigma=noise_sigma, seed=real_seed)
        rows.append(
            {
                "target_index": i,
                "source_dl1": dl[0],
                "source_dl2": dl[1],
                "source_dl3": dl[2],
                "target_x": target[0],
                "target_y": target[1],
                "target_z": target[2],
            }
        )
    return pd.DataFrame(rows)


def metrics_from_errors(df: pd.DataFrame, method: str) -> dict:
    sub = df[df["method"] == method]
    err = sub[["err_x", "err_y", "err_z"]].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(err, axis=1)
    return {
        "method": method,
        "N": int(len(sub)),
        "MAE_norm": float(np.mean(norm)),
        "RMSE_norm": float(np.sqrt(np.mean(norm**2))),
        "MAX_norm": float(np.max(norm)),
        "MAE_X": float(np.mean(np.abs(err[:, 0]))),
        "MAE_Y": float(np.mean(np.abs(err[:, 1]))),
        "MAE_Z": float(np.mean(np.abs(err[:, 2]))),
        "RMSE_X": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "RMSE_Y": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "RMSE_Z": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "MAX_abs_X": float(np.max(np.abs(err[:, 0]))),
        "MAX_abs_Y": float(np.max(np.abs(err[:, 1]))),
        "MAX_abs_Z": float(np.max(np.abs(err[:, 2]))),
    }


def run_phase4_benchmark(
    cfg: IKConfig,
    n_targets: int,
    seed: int,
    noise_sigma: float,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if n_targets < 50:
        raise ValueError("Phase 4 benchmark requires at least 50 targets.")

    out_dir.mkdir(parents=True, exist_ok=True)
    corrector = Phase3NNCorrector(
        model_path=cfg.model_path,
        x_scaler_path=cfg.x_scaler_path,
        y_scaler_path=cfg.y_scaler_path,
        device=cfg.device,
    )
    methods: list[tuple[str, ForwardFn]] = [
        ("PCC IK", pcc_forward_xyz),
        ("PCC+NN IK", make_pcc_nn_forward(corrector)),
    ]

    targets = generate_targets(n_targets, seed=seed, noise_sigma=noise_sigma)
    result_rows = []

    for _, target_row in targets.iterrows():
        target = target_row[["target_x", "target_y", "target_z"]].to_numpy(dtype=np.float64)
        for method, forward_fn in methods:
            ik = inverse_kinematics(target, forward_fn, cfg)
            eval_seed = None if noise_sigma == 0 else seed + int(target_row["target_index"]) * 100 + len(result_rows)
            reached = real_forward_xyz(ik.dl, noise_sigma=noise_sigma, seed=eval_seed)
            err = reached - target
            result_rows.append(
                {
                    "target_index": int(target_row["target_index"]),
                    "method": method,
                    "source_dl1": target_row["source_dl1"],
                    "source_dl2": target_row["source_dl2"],
                    "source_dl3": target_row["source_dl3"],
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    "ik_dl1": ik.dl[0],
                    "ik_dl2": ik.dl[1],
                    "ik_dl3": ik.dl[2],
                    "ik_dl_norm": float(np.linalg.norm(ik.dl)),
                    "ik_forward_x": ik.forward_xyz[0],
                    "ik_forward_y": ik.forward_xyz[1],
                    "ik_forward_z": ik.forward_xyz[2],
                    "ik_residual_norm": ik.residual_norm,
                    "reached_x": reached[0],
                    "reached_y": reached[1],
                    "reached_z": reached[2],
                    "err_x": err[0],
                    "err_y": err[1],
                    "err_z": err[2],
                    "err_norm": float(np.linalg.norm(err)),
                    "active_tendons": " ".join(str(i + 1) for i in ik.active_tendons),
                    "active_set": " ".join(str(i + 1) for i in ik.active_set),
                    "solver": ik.solver,
                    "nfev": ik.nfev,
                    "success": ik.success,
                }
            )

    results = pd.DataFrame(result_rows)
    metrics = pd.DataFrame([metrics_from_errors(results, method) for method, _ in methods])

    results.to_csv(out_dir / "phase4_results.csv", index=False)
    metrics.to_csv(out_dir / "comparison_metrics.csv", index=False)
    save_plots(results, out_dir)

    summary = {
        "phase": "phase4_inverse_kinematics",
        "real_model_used_inside_ik": False,
        "methods_compared": ["PCC IK", "PCC+NN IK"],
        "n_targets": int(n_targets),
        "target_generation": {
            "seed": int(seed),
            "valid_dl_max_active_tendons": 2,
            "noise_sigma": float(noise_sigma),
            "deterministic": bool(noise_sigma == 0),
        },
        "constraints": {
            "dl_min": cfg.dl_min,
            "dl_max": cfg.dl_max,
            "max_active_tendons": cfg.max_active_tendons,
        },
        "solver": {
            "preferred": "scipy.optimize.least_squares",
            "scipy_available": least_squares is not None,
            "fallback": "numerical Jacobian + damped least squares",
            "grid_n": cfg.grid_n,
            "n_starts_per_mode": cfg.n_starts_per_mode,
            "max_nfev": cfg.max_nfev,
        },
        "canonical_phase3_model": {
            "model_path": str(cfg.model_path),
            "x_scaler_path": str(cfg.x_scaler_path),
            "y_scaler_path": str(cfg.y_scaler_path),
            "device": cfg.device,
        },
        "real_model_parameters": {**REAL_MODEL_PARAMS, "sigma_noise": float(noise_sigma)},
        "outputs": {
            "results_csv": str(out_dir / "phase4_results.csv"),
            "metrics_csv": str(out_dir / "comparison_metrics.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "error_hist_png": str(out_dir / "error_hist.png"),
            "target_vs_reached_3d_png": str(out_dir / "target_vs_reached_3d.png"),
            "error_vs_target_index_png": str(out_dir / "error_vs_target_index.png"),
            "error_vs_dl_norm_png": str(out_dir / "error_vs_dl_norm.png"),
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results, metrics, summary


def save_plots(results: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    for method, sub in results.groupby("method"):
        plt.hist(sub["err_norm"], bins=30, alpha=0.55, label=method)
    plt.xlabel("||REAL(dl_ik) - target|| [mm]")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_hist.png", dpi=200)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    first = results.drop_duplicates("target_index")
    ax.scatter(first["target_x"], first["target_y"], first["target_z"], s=16, c="black", label="target")
    for method, sub in results.groupby("method"):
        ax.scatter(sub["reached_x"], sub["reached_y"], sub["reached_z"], s=10, alpha=0.65, label=method)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "target_vs_reached_3d.png", dpi=220)
    plt.close()

    plt.figure()
    for method, sub in results.groupby("method"):
        ordered = sub.sort_values("target_index")
        plt.plot(ordered["target_index"], ordered["err_norm"], marker="o", markersize=3, linewidth=1, label=method)
    plt.xlabel("target index")
    plt.ylabel("||error|| [mm]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_target_index.png", dpi=200)
    plt.close()

    plt.figure()
    for method, sub in results.groupby("method"):
        plt.scatter(sub["ik_dl_norm"], sub["err_norm"], s=14, alpha=0.7, label=method)
    plt.xlabel("||dl_ik|| [mm]")
    plt.ylabel("||error|| [mm]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_dl_norm.png", dpi=200)
    plt.close()


def solve_single_target(target: Iterable[float], cfg: IKConfig) -> dict:
    corrector = Phase3NNCorrector(
        model_path=cfg.model_path,
        x_scaler_path=cfg.x_scaler_path,
        y_scaler_path=cfg.y_scaler_path,
        device=cfg.device,
    )
    methods = [
        ("PCC IK", pcc_forward_xyz),
        ("PCC+NN IK", make_pcc_nn_forward(corrector)),
    ]
    return {
        "phase": "phase4_inverse_kinematics_single_target",
        "real_model_used_inside_ik": False,
        "methods_compared": ["PCC IK", "PCC+NN IK"],
        "target_xyz": list(map(float, target)),
        "results": {
            method: inverse_kinematics(target, forward_fn, cfg).to_dict()
            for method, forward_fn in methods
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 inverse kinematics comparison for PCC IK vs PCC+NN IK."
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional single target tip position in mm. If omitted, runs the 50+ target benchmark.",
    )
    parser.add_argument("--n-targets", type=int, default=50, help="Benchmark target count, minimum 50.")
    parser.add_argument("--seed", type=int, default=42, help="Target generation seed.")
    parser.add_argument(
        "--real-noise-sigma",
        type=float,
        default=0.0,
        help="Noise sigma used for target generation and final real_model evaluation. Default 0 for deterministic Phase 4.",
    )
    parser.add_argument("--grid-n", type=int, default=9, help="Coarse grid points per active tendon.")
    parser.add_argument("--starts", type=int, default=3, help="Best grid seeds refined per active set.")
    parser.add_argument("--max-nfev", type=int, default=120, help="Max scipy least_squares evaluations per start.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Benchmark output directory.")
    parser.add_argument("--json", action="store_true", help="Print JSON for single-target mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = IKConfig(
        grid_n=args.grid_n,
        n_starts_per_mode=args.starts,
        max_nfev=args.max_nfev,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.target is not None:
        result = solve_single_target(args.target, cfg)
        if args.json:
            print(json.dumps(result, indent=2))
            return

        print("Phase 4 inverse kinematics: single target")
        print(f"real_model.py used inside IK: {result['real_model_used_inside_ik']}")
        print(f"methods compared: {', '.join(result['methods_compared'])}")
        print(f"target XYZ [mm]: {result['target_xyz']}")
        for method, payload in result["results"].items():
            print("")
            print(method)
            print(f"  dl [mm]           : {payload['dl']}")
            print(f"  forward XYZ [mm]  : {payload['forward_xyz']}")
            print(f"  residual norm [mm]: {payload['residual_norm']:.6f}")
            print(f"  active tendons    : {payload['active_tendons']}")
            print(f"  solver            : {payload['solver']}")
        return

    results, metrics, summary = run_phase4_benchmark(
        cfg=cfg,
        n_targets=args.n_targets,
        seed=args.seed,
        noise_sigma=args.real_noise_sigma,
        out_dir=args.out_dir,
    )
    print("Phase 4 benchmark complete")
    print(f"Saved results: {summary['outputs']['results_csv']}")
    print(f"Saved metrics: {summary['outputs']['metrics_csv']}")
    print(f"Saved summary: {summary['outputs']['summary_json']}")
    print("")
    print(metrics[["method", "MAE_norm", "RMSE_norm", "MAX_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()
