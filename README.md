# Continuum Robot Kinematics · Error Compensation

**A research dashboard and reproducible study of analytical kinematics, residual learning, and physical endpoint measurements.**

This diploma project investigates whether data-driven residual models can improve the forward position predicted by a piecewise constant-curvature (PCC) model of a three-tendon continuum robot. The repository separates a synthetic reference study from experiments on measured physical endpoints. Its desktop dashboard lets you explore both without presenting simulated shapes as physical observations.

> **Measurement boundary:** Phase 6 contains measured **tip positions**, not measured robot backbones. Physical XYZ is provisionally treated as millimetres. PCC is aligned to measurements using training data; independent sensor-frame calibration has not been established.

## Dashboard

The PySide6 / pyqtgraph application has an interactive 3D analytical backbone, three live projections, synchronized tendon controls, camera presets, endpoint error vectors, a measured test-point cloud, and a results dashboard. Select an experiment to load its saved registration, scalers and model artifacts. Missing files are reported in the interface.

**Application screenshot:** pending a manual capture of the new GUI. Launch the application, use **Capture** or `Ctrl+Shift+S`, and save an actual PNG to [`figures/gui/`](figures/gui/README.md). No interface image is fabricated or linked before capture.

| Explore | What is shown |
|---|---|
| **Synthetic demonstration** | PCC curve, synthetic model curve and endpoint, saved Phase 3 NN endpoint correction. The reference is deterministic simulation output. |
| **Physical measurements** | Registered PCC analytical curve, saved Phase 6 v2 endpoint predictors, and actual endpoint observations from the selected **test split**. No measured curve is drawn. |
| **Experimental results** | Saved test RMSE, error CDF, sparse-grid and held-out-region comparisons, and per-axis errors. No benchmark is run by the GUI. |

## Methodology

The learned models predict a Cartesian endpoint residual, which is added to the analytical position:

`p_corrected(Δl) = p_PCC(Δl) + f_ML(Δl)`

![Synthetic residual-compensation workflow](figures/methodology/pipeline_synthetic_compensation.png)

Phase 6 uses a different target: a physical measured endpoint. Every experimental split fits PCC-to-measurement registration on **training** positions, selects a registration and model settings on **validation** positions, and scores only on the untouched **test** positions. Details: [methodology](docs/methodology.md) · [physical protocol](docs/physical_validation.md).

## Six phases

| Phase | Research step | Evidence |
|---|---|---|
| 1 | Analytical PCC forward kinematics and workspace | [PCC implementation](src/continuum_robot/kinematics/pcc_model.py) |
| 2 | Controlled **synthetic** reference and dataset | [Reference model](src/continuum_robot/kinematics/synthetic_reference.py) |
| 3 | Residual learning and synthetic forward validation | [Phase 3 results](results/final/phase3_forward_compensation/phase3_metrics.csv) |
| 4 | Compensated inverse kinematics on synthetic targets | [Phase 4 results](results/final/phase4_ik_validation/phase4_ik_metrics.csv) |
| 5 | Synthetic size, noise, range and architecture studies | [Summer studies](results/final/summer_experiments/) |
| 6 | Forward-model evaluation on **physical endpoint measurements** | [Reviewed v2 results](results/final/phase6_review_v2/all_metrics.csv) |

## Key results

All errors below are Euclidean test-position RMSE in **mm**. Synthetic and physical experiments use different targets and must not be compared as the same benchmark.

| Setting | Analytical baseline | Learned correction | RMSE [mm] |
|---|---:|---|---:|
| Phase 3 synthetic forward, 3,000 held-out points | PCC **2.227** | Polynomial Ridge / NN | **0.868 / 0.869** |
| Phase 4 synthetic deterministic IK, 50 targets | PCC IK **2.107** | PCC+NN IK | **0.057** |
| Phase 6 v2 physical random test, 4,538 endpoints | Train-registered PCC **4.970** | KNN / FFNN | **0.320 / 0.389** |

The Phase 6 v2 physical results show different strengths across evaluation protocols:

| Physical test | Registered PCC | KNN | FFNN | Lower RMSE |
|---|---:|---:|---:|---|
| Random interpolation | 4.970 | **0.320** | 0.389 | KNN |
| Sparse grid · 0.2 mm | 4.950 | **0.407** | 0.537 | KNN |
| Sparse grid · 0.5 mm | 4.960 | **0.577** | 0.671 | KNN |
| Sparse grid · 1.0 mm | 4.972 | **0.863** | 0.964 | KNN |
| Excluded region · plane 1 | 3.016 | 0.684 | **0.228** | FFNN |
| Excluded region · plane 2 | 4.234 | 0.656 | **0.251** | FFNN |
| Excluded region · plane 3 | 3.221 | 0.622 | **0.323** | FFNN |

Values are rounded from [v2 metrics](results/final/phase6_review_v2/all_metrics.csv). The [paired v1–v2 comparison](results/final/phase6_review_v2/paired_legacy_comparison.csv) uses identical test IDs and shows that v2 does **not** universally outperform historical v1. Full MAE, median, P95, maximum and axiswise errors remain in the CSV files. [Research answers](results/final/phase6_review_v2/research_answers.md) discuss RQ1–RQ5.

## Selected figures

| Synthetic forward validation | Physical sparse-grid generalization |
|---|---|
| ![Synthetic PCC, reference and corrected endpoints](results/final/phase3_forward_compensation/reference_pcc_nn_3d.png) | ![Physical sparse-grid RMSE](results/final/phase6_review_v2/sparse_grid_accuracy.png) |

| Physical test error distribution | Physical excluded-region test |
|---|---|
| ![Phase 6 v2 random-test errors](results/final/phase6_review_v2/random/error_distribution.png) | ![Phase 6 v2 region holdout errors](results/final/phase6_review_v2/region_2/held_out_region.png) |

More figures: [synthetic validation](docs/synthetic_validation.md) · [physical validation](docs/physical_validation.md).

## Install and run

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m continuum_robot.gui.pcc_gui
```

If an existing `.venv` already has the dependencies and editable install, use only the final command. The GUI requires a working OpenGL driver and `PyOpenGL`; it does not start any training or experiment scripts. Controls: drag to rotate, wheel to zoom, middle drag to pan, `R` straight configuration, `F` fit view, `Ctrl+Shift+S` capture an actual screenshot. [Dashboard guide](docs/reproducibility.md#dashboard-workflow).

## Reproducing experiments

The saved results are ready for inspection. To check the physical source and split lineage **without retraining**:

```powershell
.\.venv\Scripts\python.exe -m continuum_robot.phase5.review_v2 check
```

Training a fresh v2 run is a separate, explicitly invoked operation. Its command, output paths and comparison procedure are in [reproducibility](docs/reproducibility.md). Historical synthetic commands and detailed provenance remain in the [archived technical README](docs/legacy_readme.md).

## Repository map

```text
src/continuum_robot/  PCC, synthetic reference, residual models, IK, GUI
data/real/            Immutable physical source and documented preprocessing
artifacts/            Saved scalers and model weights
results/final/        Synthetic studies, historical physical v1, reviewed v2
figures/              Methodology figures and manually captured GUI screenshots
docs/                 Scientific method, validation and reproduction notes
```

## Limits and documentation

The physical dataset has 30,250 unique endpoint observations, mostly from two-tendon actuation planes. It does not provide observed backbone shapes, repeated trials, motion history, a confirmed measurement frame or three-active-tendon coverage. The empirical registration may absorb some PCC geometry error. Numeric differences between models on one static dataset do not alone establish statistical significance, and Phase 6 does not validate physical inverse kinematics.

- [Methodology and display semantics](docs/methodology.md)
- [Synthetic validation and provenance](docs/synthetic_validation.md)
- [Physical validation and coordinate assumptions](docs/physical_validation.md)
- [Install, dashboard use and intentional reruns](docs/reproducibility.md)
- [Archived detailed README](docs/legacy_readme.md)
