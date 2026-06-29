# Phase 3: NN for PCC error compensation (ΔL → ΔX,ΔY,ΔZ)
# Outputs: model, scalers, metrics, loss plot

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FALLBACK_REAL_MODEL_PARAMS = {
    "alpha_per_m": 3.5,
    "beta_rad_per_m": 3.5,
    "offset": [0.5, 0.5, 0.3],
    "sigma_noise": 0.5,
    "theta_max_deg": 95.0,
}


def gaussian_noise_floor_norm(real_model_params: dict) -> float:
    sigma = float(real_model_params["sigma_noise"])
    return float(np.sqrt(3.0) * sigma)


def load_dataset_provenance(data_path: Path) -> dict:
    provenance_path = data_path.with_name("dataset_3_provenance.json")
    if not provenance_path.exists():
        return {
            "path": str(provenance_path),
            "available": False,
            "real_model_parameters": FALLBACK_REAL_MODEL_PARAMS,
            "estimated_gaussian_noise_floor_rmse_norm_mm": gaussian_noise_floor_norm(FALLBACK_REAL_MODEL_PARAMS),
        }

    with open(provenance_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)
    provenance["path"] = str(provenance_path)
    provenance["available"] = True

    real_params = provenance.get("real_model_parameters", FALLBACK_REAL_MODEL_PARAMS)
    provenance.setdefault("real_model_parameters", real_params)
    provenance.setdefault(
        "estimated_gaussian_noise_floor_rmse_norm_mm",
        gaussian_noise_floor_norm(real_params),
    )
    return provenance


# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    data_path: Path = Path("dataset_out/dataset_3.npz")
    out_dir: Path = Path("phase3_last_16_03_2026")

    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15

    batch_size: int = 256
    epochs: int = 300
    lr: float = 0.00019591496594730567
    weight_decay: float = 2.5465418104826876e-08

    # Feature options
    use_activity_mask: bool = True
    eps_activity: float = 1e-9

    hidden_sizes: tuple[int, ...] = (215, 68)
    activation: str = "relu"

    # Loss: MSE or MAE
    loss: str = "mse"
    patience: int = 30
    min_delta: float = 1e-5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Dataset wrapper
# -----------------------
class NumpyXYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# -----------------------
# Model
# -----------------------
def get_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be 'relu' or 'tanh'")


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
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
        layers.append(nn.Linear(prev, out_dim))  # linear output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------
# Metrics
# -----------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def vec_norm_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Per-sample Euclidean error
    d = a - b
    return np.sqrt(np.sum(d * d, axis=1))


# -----------------------
# Train / Eval
# -----------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    # Evaluate model (no grad)
    model.eval()
    Y_true_list = []
    Y_pred_list = []
    for Xb, Yb in loader:
        Xb = Xb.to(device)
        pred = model(Xb)
        Y_true_list.append(Yb.cpu().numpy())
        Y_pred_list.append(pred.cpu().numpy())
    return np.vstack(Y_true_list), np.vstack(Y_pred_list)


def train(cfg: Config):
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_provenance = load_dataset_provenance(cfg.data_path)

    # ---- Load NPZ (X,Y)
    data = np.load(cfg.data_path)
    if "X" not in data or "Y" not in data:
        raise KeyError("NPZ must contain 'X' and 'Y'")

    X = data["X"].astype(np.float64)
    Y = data["Y"].astype(np.float64)

    # Optional activity mask
    if cfg.use_activity_mask:
        A = (np.abs(X) > cfg.eps_activity).astype(np.float64)
        X = np.hstack([X, A])  # (N,6)

    if X.ndim != 2 or Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError(f"Bad shapes: X{X.shape}, Y{Y.shape}")

    # ---- Split train/val/test
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
    )
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_rel, random_state=cfg.seed, shuffle=True
    )

    # ---- Standardization (fit on train only)
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_n = x_scaler.transform(X_train)
    X_val_n = x_scaler.transform(X_val)
    X_test_n = x_scaler.transform(X_test)

    Y_train_n = y_scaler.transform(Y_train)
    Y_val_n = y_scaler.transform(Y_val)
    Y_test_n = y_scaler.transform(Y_test)

    # ---- DataLoaders
    train_loader = DataLoader(NumpyXYDataset(X_train_n, Y_train_n),
                              batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(NumpyXYDataset(X_val_n, Y_val_n),
                            batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(NumpyXYDataset(X_test_n, Y_test_n),
                             batch_size=cfg.batch_size, shuffle=False)

    # ---- Model
    in_dim = X_train_n.shape[1]
    model = FeedForwardNN(in_dim=in_dim,
                          hidden_sizes=cfg.hidden_sizes,
                          activation=cfg.activation).to(cfg.device)

    # ---- Loss (MSE or MAE)
    loss_name = cfg.loss.lower().strip()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    # ---- Training with early stopping
    best_val = float("inf")
    best_path = cfg.out_dir / "nn_model.pt"
    x_scaler_path = cfg.out_dir / "x_scaler.pkl"
    y_scaler_path = cfg.out_dir / "y_scaler.pkl"
    history = {"train_loss": [], "val_loss": []}
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for Xb, Yb in train_loader:
            Xb = Xb.to(cfg.device)
            Yb = Yb.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(cfg.device)
                Yb = Yb.to(cfg.device)
                pred = model(Xb)
                val_losses.append(criterion(pred, Yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping check
        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            patience_left = cfg.patience

            cfg_dict = dict(cfg.__dict__)
            cfg_dict["in_dim"] = int(in_dim)
            cfg_dict["loss"] = loss_name
            for k in ["data_path", "out_dir"]:
                if k in cfg_dict and cfg_dict[k] is not None:
                    cfg_dict[k] = str(cfg_dict[k])

            torch.save(
                {"model_state": model.state_dict(),
                 "config": cfg_dict},
                best_path,
            )
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | "
                  f"val {val_loss:.6f} | best {best_val:.6f} | "
                  f"patience {patience_left}")

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    # ---- Load best model
    ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Test evaluation (denormalized)
    Yt_n, Yp_n = eval_model(model, test_loader, cfg.device)
    Yt = y_scaler.inverse_transform(Yt_n)
    Yp = y_scaler.inverse_transform(Yp_n)

    e_norm = vec_norm_err(Yt, Yp)

    metrics = {
        "dataset_path": str(cfg.data_path),
        "model_path": str(best_path),
        "x_scaler_path": str(x_scaler_path),
        "y_scaler_path": str(y_scaler_path),
        "seed": int(cfg.seed),
        "split_sizes": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "real_model_parameters": dataset_provenance["real_model_parameters"],
        "estimated_gaussian_noise_floor_rmse_norm_mm": dataset_provenance[
            "estimated_gaussian_noise_floor_rmse_norm_mm"
        ],
        "dataset_provenance": dataset_provenance,
        "MAE_xyz": mae(Yt, Yp),
        "RMSE_xyz": rmse(Yt, Yp),
        "MAX_abs_xyz": max_err(Yt, Yp),
        "MAE_norm": float(np.mean(e_norm)),
        "RMSE_norm": float(np.sqrt(np.mean(e_norm ** 2))),
        "MAX_norm": float(np.max(e_norm)),
        "N_train": int(X_train.shape[0]),
        "N_val": int(X_val.shape[0]),
        "N_test": int(X_test.shape[0]),
        "loss": loss_name,
        "activation": cfg.activation.lower(),
        "hidden_sizes": list(cfg.hidden_sizes),
        "use_activity_mask": bool(cfg.use_activity_mask),
        "in_dim": int(in_dim),
    }

    # ---- Save artifacts
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    with open(cfg.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "loss_curve.png", dpi=200)
    plt.close()

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"{k:>16s}: {v}")

    print(f"\nSaved to: {cfg.out_dir.resolve()}")
    return metrics


# -----------------------
# Inference helper
# -----------------------
@torch.no_grad()
def predict_delta_xyz(
    dl: np.ndarray,
    model_path: Path,
    x_scaler_path: Path,
    y_scaler_path: Path,
    device: str | None = None,
) -> np.ndarray:
    """
    dl: (N,3)
    returns: (N,3) ΔX,ΔY,ΔZ
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_scaler: StandardScaler = joblib.load(x_scaler_path)
    y_scaler: StandardScaler = joblib.load(y_scaler_path)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    hidden_sizes = tuple(cfg_dict.get("hidden_sizes", (64, 64)))
    activation = cfg_dict.get("activation", "relu")
    use_activity_mask = bool(cfg_dict.get("use_activity_mask", True))
    eps_activity = float(cfg_dict.get("eps_activity", 1e-9))
    in_dim = int(cfg_dict.get("in_dim", 3))

    model = FeedForwardNN(in_dim=in_dim,
                          hidden_sizes=hidden_sizes,
                          activation=activation).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dl = np.asarray(dl, dtype=np.float64)
    if dl.ndim == 1:
        dl = dl.reshape(1, 3)
    if dl.shape[1] != 3:
        raise ValueError(f"dl must be (N,3), got {dl.shape}")

    if use_activity_mask:
        A = (np.abs(dl) > eps_activity).astype(np.float64)
        dl = np.hstack([dl, A])

    if dl.shape[1] != in_dim:
        raise ValueError("Feature dim mismatch")

    dl_n = x_scaler.transform(dl).astype(np.float32)
    pred_n = model(torch.from_numpy(dl_n).to(device)).cpu().numpy()
    return y_scaler.inverse_transform(pred_n)


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
