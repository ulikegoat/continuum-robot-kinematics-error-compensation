"""Run physical-data Phase 6 with training-only empirical coordinate registration.

This is an identified PCC baseline. It does not claim an externally known sensor frame.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_robot.phase5 import pipeline as p


def fit_rigid(source: np.ndarray, target: np.ndarray, orientation: int):
    a = source - source.mean(0); b = target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    diagonal = np.diag([1., 1., orientation * np.linalg.det(u @ vt)])
    row_rotation = u @ diagonal @ vt
    translation = target.mean(0) - source.mean(0) @ row_rotation
    return row_rotation, translation


def register_experiment(name: str, data: Path = p.DATA):
    split = json.loads((p.RESULTS / "splits" / f"{name}.json").read_text())
    if split["basis"] != "real_dataset_flagged.csv" or split["dataset_sha256"] != p.digest(data / "real_dataset_flagged.csv"):
        raise ValueError("Recreate source-only splits before empirical registration")
    frame = pd.read_csv(data / "real_dataset_flagged.csv")
    frame = frame[frame.is_valid].set_index("row_id", drop=False)
    commands = frame[p.INPUTS].to_numpy(float)
    target = frame[p.XYZ].to_numpy(float)
    train = frame.index.get_indexer(split["train"])
    val = frame.index.get_indexer(split["validation"])
    if (train < 0).any() or (val < 0).any(): raise ValueError("Split contains unknown source row")
    trials = []
    for order in itertools.permutations(range(3)):
        local = p.pcc_xyz(commands[:, order] / 10)
        for orientation in (-1, 1):
            rotation, translation = fit_rigid(local[train], target[train], orientation)
            validation_rmse = p.error_metrics(local[val] @ rotation + translation - target[val])["RMSE_norm"]
            trials.append((validation_rmse, order, orientation, rotation, translation, local))
    score, order, orientation, rotation, translation, local = min(trials, key=lambda row: row[0])
    predicted = local @ rotation + translation
    for i, col in enumerate(p.PCC): frame[col] = predicted[:, i]
    frame[p.RESIDUAL] = target - predicted
    frame["e_pcc"] = np.linalg.norm(frame[p.RESIDUAL].to_numpy(), axis=1)
    folder = data / "experiments" / name; folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "real_vs_pcc.csv", index=False)
    provenance = {"registration_method": "train_only_empirical", "calibration": {"verified": False},
        "input_sha256": p.digest(data / "real_dataset_flagged.csv"),
        "prediction_sha256": p.digest(folder / "real_vs_pcc.csv"),
        "source_units_assumed": "mm", "scale_fit": False,
        "PCC_geometry_assumed_mm": {"L": 110, "d": 5},
        "tendon_order_selected": list(order), "orientation_determinant": orientation,
        "row_rotation": rotation.tolist(), "translation_mm": translation.tolist(),
        "candidate_validation_RMSE_mm": [{"order": list(item[1]), "orientation": item[2], "RMSE": item[0]} for item in trials],
        "selected_validation_RMSE_mm": score,
        "registration_fit_ids": split["train"], "selection_ids": split["validation"],
        "test_ids_not_used_for_registration": split["test"],
        "warning": "Rigid registration fitted from observed physical positions in train only; this identified PCC baseline differs from uncalibrated PCC. No physical sensor-frame information was available."}
    p.write_json(folder / "prediction_provenance.json", provenance)
    frame[p.INPUTS + p.XYZ + p.PCC + p.RESIDUAL + ["e_pcc"]].to_csv(folder / "residual_dataset.csv", index=True)
    print(f"registered {name}: order={order}, orientation={orientation}, validation RMSE={score:.3f} mm", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", choices=["all", "random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3"], default="all")
    args = parser.parse_args()
    if args.epochs < 1: parser.error("epochs must be positive")
    source = p.DATA / "Dataset-Actions-Positions.txt"
    if not source.exists(): parser.error(f"Missing source: {source}")
    analysis_path = p.DATA / "dataset_analysis.json"
    analysis_current = analysis_path.exists() and json.loads(analysis_path.read_text())["source_sha256"] == p.digest(source)
    if not analysis_current: p.analyze(source)
    split_path = p.RESULTS / "splits" / "random.json"
    split_current = split_path.exists() and json.loads(split_path.read_text())["dataset_sha256"] == p.digest(p.DATA / "real_dataset_flagged.csv") and json.loads(split_path.read_text())["seed"] == args.seed
    if not split_current: p.prepare(seed=args.seed)
    names = ["random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3"] if args.experiment == "all" else [args.experiment]
    for name in names:
        completed = all((p.MODELS / name / variant / "metadata.json").exists() for variant in ("all", "clean_train"))
        completed &= (p.RESULTS / name / "metrics.csv").exists() and (p.RESULTS / name / "plane_error_maps.png").exists()
        if completed:
            completed &= all(json.loads((p.MODELS / name / variant / "metadata.json").read_text())["nn"]["max_epochs"] == args.epochs for variant in ("all", "clean_train"))
        if completed:
            print(f"reusing completed {name}", flush=True)
            continue
        folder = register_experiment(name)
        p.train_experiment(name, data=folder, epochs=args.epochs, seed=args.seed)
        p.evaluate_experiment(name, data=folder)
        original = p.DATA
        try:
            p.DATA = folder
            p.plot_experiment(name)
        finally:
            p.DATA = original
    if args.experiment == "all": p.summarise()


if __name__ == "__main__": main()
