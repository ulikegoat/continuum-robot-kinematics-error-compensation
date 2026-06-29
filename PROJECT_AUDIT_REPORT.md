# Project Audit Report

Audit date: 2026-06-29  
Repository: `D:\workspace_dp`

## Scope

This audit reviewed the active project scripts, saved datasets, saved model artifacts, metrics tables, and legacy `old/` contents. No source code was modified, no model was retrained, and no dataset was regenerated. The only intended write is this report.

Primary files reviewed:

- `generate_dataset.py`
- `csv_to_nzp.py`
- `pcc_model.py`
- `real_model.py`
- `ml_models.py`
- `train_nn_optuna.py`
- `train_nn_phase3.py`
- `eval_phase3_final.py`
- `pcc_gui.py`
- `README.md`
- saved artifacts under `dataset_out/`, `models/`, `phase3_version_3/`, `phase3_last_16_03_2026/`, `phase3_final/`, and `optuna_results/`

## Executive Summary

The repository contains enough code and artifacts to demonstrate synthetic PCC error compensation, but it is not yet Phase 4 ready. The main blocker is not a single bug; it is that the project currently has several incompatible experiment lineages:

- dataset generation currently writes `dataset_out/dataset_3.csv`;
- CSV-to-NPZ conversion writes `dataset_out/dataset_3.npz`;
- classical models train from `dataset_out/dataset.csv`;
- current NN training writes to `phase3_last_16_03_2026`;
- final evaluation uses `dataset_out/dataset_phase2_synth_new.npz` and `phase3_version_3`;
- the GUI loads only sklearn models from `models/`, not the final NN.

The final Phase 3 numbers are also very close to the expected label-noise floor. The simulated real model adds independent Gaussian noise to each output axis, and the reported compensated `RMSE_norm` values near 0.86 mm are approximately `sqrt(3) * 0.5 mm`. That means the current results mostly show that the deterministic component has been learned and the remaining error is dominated by injected noise, not necessarily that the NN is better than the polynomial baseline.

## High Priority Findings

### 1. No Single Canonical Dataset/Model/Evaluation Path

There are multiple competing data and artifact paths:

| Component | Current path in code/artifact | Evidence |
|---|---:|---|
| Dataset generator CSV | `dataset_out/dataset_3.csv` | `generate_dataset.py:17-19` |
| CSV-to-NPZ conversion | `dataset_out/dataset_3.npz` | `csv_to_nzp.py:5-18` |
| Classical model training | `dataset_out/dataset.csv` | `ml_models.py:16` |
| Current NN training input/output | `dataset_out/dataset_3.npz`, `phase3_last_16_03_2026` | `train_nn_phase3.py:26-27` |
| Final evaluation input | `dataset_out/dataset_phase2_synth_new.npz` | `eval_phase3_final.py:36` |
| Final evaluation NN artifact | `phase3_version_3` | `eval_phase3_final.py:39-42` |
| GUI model source | sklearn `.pkl` files from `models/` | `pcc_gui.py:22`, `pcc_gui.py:414-486` |

Saved dataset inventory confirms this split:

- `dataset.csv`: 30,000 rows
- `dataset_3.csv` / `dataset_3.npz`: 20,000 rows
- `dataset_new.csv` / `dataset_phase2_synth_new.npz`: 100,000 rows
- `dataset_phase2_synth.npz`: 30,000 rows

Impact: final metrics, GUI behavior, README workflow, and current training script do not refer to the same experiment. A Phase 4 user cannot tell which model is the approved one.

### 2. Saved `phase3_final` Artifacts Do Not Match the Current Evaluation Script Exactly

The saved `phase3_final/summary.json` contains fields such as:

- `boundary: { "thr": ..., "rule": ... }`
- `polydeg3: { ..., "retrained_on": "NPZ train split" }`

The current `eval_phase3_final.py` writes:

- `boundary_thr`
- `polydeg3` without `retrained_on`

Evidence: `eval_phase3_final.py:371-386`.

Impact: the checked-in final evaluation outputs appear to have been produced by an older script revision. This weakens traceability and makes the reported final figures hard to reproduce exactly from the current repo state.

### 3. The Final Reported Error Is Essentially the Synthetic Noise Floor

`generate_dataset.py` configures:

- `real.sigma_noise = 0.5` at `generate_dataset.py:27`
- the real model adds `np.random.normal(0.0, sigma_noise, size=3)` at `real_model.py:105-106`

For independent 0.5 mm Gaussian noise on X/Y/Z, the irreducible `RMSE_norm` floor is approximately:

```text
sqrt(3 * 0.5^2) = 0.866 mm
```

The saved final results are:

- `PCC + PolyDeg3`: `RMSE_norm = 0.8613`
- `PCC + NN`: `RMSE_norm = 0.8631`

Impact: the final comparison is dominated by injected measurement noise. The NN is not meaningfully better than PolyDeg3 on the final test set, and both are approximately at the expected noise floor. This should be explicitly stated in any thesis/Phase 4 claims.

### 4. Dataset Generation Is Not Fully Reproducible

`generate_dataset.py` seeds only the local `np.random.default_rng(SEED)` used for tendon shortening samples at `generate_dataset.py:94-95`. The Gaussian noise added in `real_model.real_forward()` uses global `np.random.normal()` at `real_model.py:105-106`, and that global RNG is not seeded by `generate_dataset.py`.

Impact: rerunning `generate_dataset.py` with the same `SEED` will reproduce input `dl` samples, but not the noisy real outputs or labels. This affects reproducibility of model metrics.

### 5. README Workflow Is Incomplete and Points to a Mixed Experiment

The README instructs:

1. `python generate_dataset.py`
2. `python train_nn_phase3.py`
3. `python eval_phase3_final.py`

Evidence: `README.md:40-55`.

Problems:

- `generate_dataset.py` creates only `dataset_3.csv`.
- `train_nn_phase3.py` requires `dataset_3.npz`.
- the required conversion step `python csv_to_nzp.py` is missing.
- `eval_phase3_final.py` does not evaluate the `phase3_last_16_03_2026` model produced by current training.

Impact: a fresh user following the README cannot reproduce the final evaluation path cleanly.

## Methodological Findings

### 6. The "REAL" Model Is Simulated, Not Real Experimental Ground Truth

`real_model.py` is a synthetic perturbation of PCC:

- curvature nonlinearity,
- plane asymmetry,
- bend saturation,
- endpoint bias,
- Gaussian noise.

Evidence: `real_model.py:4-17`, `real_model.py:40-64`, `real_model.py:95-107`.

Impact: the current project validates compensation against a simulated model, not measurements from a physical robot. Phase 4 should not present this as real-world validation without a separate measured test set.

### 7. Dataset-Generation Parameters Are Hidden Runtime Mutations

`generate_dataset.py` mutates globals in `real_model`:

- `alpha_per_m = 3.5`
- `beta_rad_per_m = 3.5`
- `offset = [0.5, 0.5, 0.3]`
- `sigma_noise = 0.5`
- `theta_max = 95 deg`

Evidence: `generate_dataset.py:22-30`.

But `real_model.py` defaults are different:

- `alpha_per_m = 0.05`
- `beta_rad_per_m = 1.746`
- `offset = [1.0, 1.0, 0.5]`
- `sigma_noise = 0.8`

Evidence: `real_model.py:20-24`.

Impact: scripts that import `real_model.py` directly, such as the GUI, do not necessarily use the same "real" model used to generate the training data.

### 8. GUI Error Display Is Not Methodologically Comparable to Dataset Metrics

The GUI uses `real.real_shape()` through `safe_real_shape()` at `pcc_gui.py:50-55` and `pcc_gui.py:477-490`. `real_shape()` does not add endpoint offset or Gaussian noise; those are only added by `real_forward()` at `real_model.py:95-107`.

Impact: GUI errors are not comparable to training labels or final evaluation metrics. The GUI visualizes a deterministic shape difference, while the datasets include noisy biased endpoints.

### 9. The GUI Does Not Use the Final NN

`pcc_gui.py` loads sklearn `.pkl` models from `models/` and predicts with `self.ml_model.predict([[l1, l2, l3]])`.

Evidence: `pcc_gui.py:22`, `pcc_gui.py:432-486`.

The final NN artifacts are `.pt` plus scalers in `phase3_version_3` or `phase3_last_16_03_2026`; the GUI does not load them.

Impact: if Phase 4 means a demonstrator or deployment UI, the deployed model is not the final evaluated NN.

### 10. Classical Model Artifacts Are Not the Same PolyDeg3 Used in Final Evaluation

`ml_models.py` trains and saves `models/PolyDeg3.pkl` from `dataset_out/dataset.csv`.

Evidence: `ml_models.py:16`, `ml_models.py:186-188`.

`eval_phase3_final.py` retrains a new PolyDeg3 on the NPZ train split at evaluation time.

Evidence: `eval_phase3_final.py:197-205`, `eval_phase3_final.py:285-288`.

Impact: `models/PolyDeg3.pkl` is not the same model as the one reported in `phase3_final`.

### 11. Boundary Subset Definition Is Partly Implicit

The boundary subset is initially:

```text
max(dl) >= 9.0 and active_cnt == 2
```

but if fewer than 200 samples match, it falls back to:

```text
max(dl) >= 9.0
```

Evidence: `eval_phase3_final.py:320-329`.

Impact: the actual boundary definition can change silently depending on dataset size/distribution. The summary does not record whether fallback was used.

### 12. PCC `theta` Value May Be Inconsistent With the Rendered Shape

`pcc_model.py` computes `theta = 2 * root / (3 * d)` at `pcc_model.py:27-28`, but `pcc_shape()` renders the curve over fixed `L = 110.0` using `kappa * s` at `pcc_model.py:43-48`.

If `theta` is intended to represent the bending angle of the rendered centerline, it should be consistent with `kappa * L`. Currently it is based on the average tendon length formula, while the rendered arc uses fixed backbone length.

Impact: this may affect displayed angles and saturation reasoning. It does not directly invalidate the learned error metrics, because the datasets use the same implementation consistently, but it should be resolved before Phase 4.

## Data Leakage and Split Integrity

### What Looks Correct

- `ml_models.py` splits test first, then validation from the remaining data, using fixed `random_state=42`. Evidence: `ml_models.py:98-107`.
- `train_nn_phase3.py` splits test first, then validation from the remaining data. Evidence: `train_nn_phase3.py:170-177`.
- `train_nn_phase3.py` fits `StandardScaler` only on training data. Evidence: `train_nn_phase3.py:179-189`.
- `eval_phase3_final.py` trains PolyDeg3 only on `X_train`, `Y_train`. Evidence: `eval_phase3_final.py:273-288`.
- Compensation sign is correct where `Y = REAL - PCC`: residual is `-Y + prediction`. Evidence: `eval_phase3_final.py:282-298`.

### Remaining Leakage/Integrity Risks

- Split indices are not saved, so artifacts cannot prove which exact samples were used for training, validation, and test.
- `eval_phase3_final.py` assumes the loaded NN was trained on the same dataset and split, but it does not verify this beyond reading checkpoint config.
- Hyperparameter tuning in `train_nn_optuna.py` repeatedly optimizes against the same validation split; this is acceptable for model selection, but final claims should rely only on the untouched test set.
- There is no external measured holdout set. All current train/test data are generated from the same synthetic process.

## Metric Correctness

### Correct or Acceptable

- Vector residual norm metrics are computed correctly as row-wise Euclidean norms. Evidence: `ml_models.py:39-53`, `eval_phase3_final.py:72-82`, `train_nn_phase3.py:119-122`.
- Axis MAE/RMSE/MAX calculations are standard. Evidence: `ml_models.py:54-62`, `eval_phase3_final.py:83-91`.
- `MAE_norm` uses `np.abs(n)`, but `n` is already nonnegative; this is redundant, not wrong.

### Metric Reporting Issues

- `train_nn_phase3.py` labels scalar axis-aggregated values as `MAE_xyz`, `RMSE_xyz`, and `MAX_abs_xyz`. Evidence: `train_nn_phase3.py:297-303`. These aggregate across all coordinates, while `MAE_norm` and `RMSE_norm` are vector-norm metrics. This should be clarified in reports.
- Final CSV/TEX tables save only norm metrics, even though axis metrics are computed. Evidence: `eval_phase3_final.py:309-318`, `eval_phase3_final.py:342-351`.
- No confidence intervals, repeated seeds, or noise-floor-normalized metrics are reported.
- Because label noise dominates the final residual, small PolyDeg3-vs-NN differences are not practically meaningful without repeated evaluation.

## Reproducibility Findings

### 13. Optuna Study Is Not Reproducible

`train_nn_optuna.py` does not seed the Optuna sampler, does not call a global seed setup before model initialization, and uses unseeded `np.random.choice()` when reducing larger datasets.

Evidence:

- `np.random.choice()` at `train_nn_optuna.py:261-267`
- `optuna.create_study(direction="minimize")` at `train_nn_optuna.py:271`
- no `set_seed()` equivalent in `main()`

Impact: `optuna_results/best_params.json` is not reliably reproducible.

### 14. Training Is Partly Seeded, But Not Fully Deterministic Across Hardware

`train_nn_phase3.py` seeds Python, NumPy, and Torch at `train_nn_phase3.py:128-134`, which is good. It does not set deterministic CUDA behavior or record library/hardware versions in metrics.

Impact: CPU runs may be mostly reproducible; CUDA runs may vary.

### 15. Dependency Versions Are Not Locked

The README installs unpinned packages:

```text
pip install numpy pandas matplotlib scikit-learn torch joblib
```

Evidence: `README.md:28-34`.

Read-only artifact introspection showed sklearn pickle warnings: saved estimators were created with scikit-learn 1.6.1, while the current `.venv` has scikit-learn 1.8.0. One loaded sklearn model representation failed with:

```text
AttributeError: 'LinearRegression' object has no attribute 'tol'
```

Impact: persisted `.pkl` models may not be reliable across environments. This is a Phase 4 deployment risk.

### 16. Model Artifacts Lack Complete Provenance

Saved metrics contain model sizes and split counts, but not:

- git commit hash,
- dataset hash,
- split indices,
- Python/package versions,
- real-model parameters,
- random seed state,
- hardware/device details beyond `device`.

Impact: results are hard to audit or regenerate after code changes.

## Path and Artifact Inventory

### Dataset Artifacts

| Artifact | Rows | Notes |
|---|---:|---|
| `dataset_out/dataset.csv` | 30,000 | Used by `ml_models.py`; older distribution |
| `dataset_out/dataset_3.csv` | 20,000 | Current generator output |
| `dataset_out/dataset_3.npz` | 20,000 | Current NN training input |
| `dataset_out/dataset_new.csv` | 100,000 | CSV counterpart to final Phase 3 dataset lineage |
| `dataset_out/dataset_phase2_synth.npz` | 30,000 | Older NN lineage |
| `dataset_out/dataset_phase2_synth_new.npz` | 100,000 | Used by final evaluation and `phase3_version_3` |

### Model/Metric Artifacts

| Artifact | Dataset lineage | Notes |
|---|---|---|
| `models/*.pkl` | `dataset_out/dataset.csv` | Classical models used by GUI |
| `phase3_version_3/nn_model.pt` | `dataset_phase2_synth_new.npz` | Used by final evaluation |
| `phase3_last_16_03_2026/nn_model.pt` | `dataset_3.npz` | Produced by current `train_nn_phase3.py` |
| `phase3_final/*` | `dataset_phase2_synth_new.npz` plus `phase3_version_3` | Saved outputs do not exactly match current eval script |
| `optuna_results/best_params.json` | unclear/reproducibility risk | Best params match current `train_nn_phase3.py`, but tuning is not deterministic |

## Phase 4 Readiness Assessment

Current status: not ready.

Blocking items before Phase 4:

1. Choose one canonical dataset and one canonical final model.
2. Align `generate_dataset.py`, `csv_to_nzp.py`, `train_nn_phase3.py`, `eval_phase3_final.py`, README, and GUI around that canonical path.
3. Regenerate final evaluation artifacts from the current code after paths are aligned.
4. Save dataset hashes, split indices, real-model parameters, dependency versions, and git commit in every metrics file.
5. Explicitly report the Gaussian noise floor and avoid claiming NN superiority when PolyDeg3 and NN are statistically indistinguishable.
6. Add an external measured validation plan if Phase 4 is meant to represent real robot readiness.
7. Integrate the intended final model into the GUI or clearly state that the GUI is a classical-model demo.
8. Pin dependencies or export a reproducible environment file.

Recommended acceptance criteria for Phase 4:

- `python generate_dataset.py`
- `python csv_to_nzp.py`
- `python train_nn_phase3.py`
- `python eval_phase3_final.py`

should run from a clean checkout and produce a self-consistent report using the same dataset lineage. The generated `summary.json` should include:

- dataset path and hash,
- model artifact path and hash,
- split indices or split hash,
- full config,
- package versions,
- real-model parameters,
- noise floor,
- all norm and per-axis metrics,
- boundary subset count and exact rule used.

## Overall Conclusion

The compensation logic and metric signs are mostly correct, and the basic train/validation/test separation is reasonable. The strongest technical result is that both PolyDeg3 and NN reduce the synthetic PCC residual to approximately the expected 0.5 mm per-axis Gaussian noise floor.

However, the repository is currently fragmented across several dataset/model generations, the saved final artifacts are not fully traceable to the current script versions, and reproducibility is incomplete. Before Phase 4, the project should be consolidated into one canonical, reproducible pipeline and evaluated either against a clearly documented synthetic benchmark or a genuinely measured robot dataset.
