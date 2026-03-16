from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
@dataclass
class Cfg:
    seed: int = 42

    dataset_path: Path = Path("dataset_out/dataset_3.npz")
    out_dir: Path = Path("optuna_results")

    max_samples: int = 20000

    test_size: float = 0.15
    val_size: float = 0.15

    max_epochs: int = 120
    patience: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Dataset
# -----------------------
class NumpyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -----------------------
# NN
# -----------------------
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError()


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, hidden_sizes, activation):
        super().__init__()

        layers = []
        prev = in_dim

        act = get_activation(activation)

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            prev = h

        layers.append(nn.Linear(prev, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------
# Metrics
# -----------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -----------------------
# Training
# -----------------------
def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    losses = []

    for X, Y in loader:

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred, Y)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):

    model.eval()

    losses = []

    for X, Y in loader:

        X = X.to(device)
        Y = Y.to(device)

        pred = model(X)

        loss = loss_fn(pred, Y)

        losses.append(loss.item())

    return np.mean(losses)


# -----------------------
# Optuna objective
# -----------------------
def build_objective(cfg, X, Y):

    def objective(trial):

        hidden1 = trial.suggest_int("hidden1", 32, 256)
        hidden2 = trial.suggest_int("hidden2", 32, 256)

        activation = trial.suggest_categorical("activation", ["relu", "tanh"])

        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

        use_activity_mask = trial.suggest_categorical("use_activity_mask", [True, False])

        # features
        feats = X

        if use_activity_mask:

            A = (np.abs(X) > 1e-9).astype(np.float64)

            feats = np.hstack([X, A])

        # split
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(
            feats, Y, test_size=cfg.test_size, random_state=cfg.seed
        )

        val_rel = cfg.val_size / (1.0 - cfg.test_size)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=val_rel, random_state=cfg.seed
        )

        # scalers
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(Y_train)

        X_train = x_scaler.transform(X_train)
        X_val = x_scaler.transform(X_val)

        Y_train = y_scaler.transform(Y_train)
        Y_val = y_scaler.transform(Y_val)

        train_loader = DataLoader(
            NumpyDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            NumpyDataset(X_val, Y_val),
            batch_size=batch_size,
        )

        model = FeedForwardNN(
            in_dim=X_train.shape[1],
            hidden_sizes=(hidden1, hidden2),
            activation=activation,
        ).to(cfg.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        loss_fn = nn.MSELoss()

        best_val = float("inf")
        patience = cfg.patience

        for epoch in range(cfg.max_epochs):

            train_epoch(model, train_loader, optimizer, loss_fn, cfg.device)

            val_loss = eval_epoch(model, val_loader, loss_fn, cfg.device)

            if val_loss < best_val:

                best_val = val_loss
                patience = cfg.patience

            else:

                patience -= 1

            if patience <= 0:
                break

        return best_val

    return objective


# -----------------------
# Main
# -----------------------
def main():

    cfg = Cfg()

    cfg.out_dir.mkdir(exist_ok=True)

    data = np.load(cfg.dataset_path)

    X = data["X"]
    Y = data["Y"]

    # reduce dataset
    if X.shape[0] > cfg.max_samples:

        idx = np.random.choice(X.shape[0], cfg.max_samples, replace=False)

        X = X[idx]
        Y = Y[idx]

    objective = build_objective(cfg, X, Y)

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=40)

    print("\nBest params:")
    print(study.best_params)

    # save params
    with open(cfg.out_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    # plots
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "optuna_history.png")
    plt.close()

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "optuna_importance.png")
    plt.close()


if __name__ == "__main__":
    main()