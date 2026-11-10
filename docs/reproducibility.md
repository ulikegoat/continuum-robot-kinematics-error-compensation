# Reproducibility and artifact use

Run commands from the repository root (`D:\workspace_dp`) in PowerShell. The dashboard reads saved model and result files; launching it never trains or regenerates data.

## Install and launch

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m continuum_robot.gui.pcc_gui
```

Python 3.11 or another supported version can be used. `PyOpenGL` is required by the `pyqtgraph.opengl` viewport. On systems where the current `.venv` already contains the dependencies and editable install, use only the final launch command.

## Dashboard workflow

1. Select **Synthetic demonstration** to compare PCC, deterministic synthetic reference, and the saved canonical Phase 3 NN. The physical dataset is not used in this mode.
2. Select **Physical measurements** and an experiment. The GUI loads that experiment's v2 registration, scalers, model weights, metrics and test predictions together. Choose PCC, KNN or FFNN (or a regression baseline).
3. Use **Go to test point** to select an actual physical endpoint observation. Moving the sliders to a command without a selected-test observation shows predictions and an explicit unavailable-measurement status.
4. Rotate the 3D view by dragging; zoom with the wheel; pan with middle drag. Use the camera presets, **Focus robot** (`F`) or **Fit workspace**. Camera framing remains stable while moving tendon sliders. Press `R` for the straight configuration.
5. In the **DISPLAY** inspector group, choose cloud **Off**, sampled **Light** (default) or **Full**. The gray-to-copper scale reports the selected model's test error and clips at its 95th percentile; the measured endpoint marker is the selected test observation. Full renders every saved test endpoint in the 3D view and may be slower on some GPUs.
6. Use the left rail to open **Robot**, **Compensation**, **Measurements** or the dedicated **Analysis** workspace. Save an actual application-only PNG using **Capture** or `Ctrl+Shift+S`, or use **Save 3D viewport PNG** for the OpenGL view alone. Save under [`figures/gui/`](../figures/gui/), inspect that the 3D scene appears, then embed the image in the main README.

## Validate saved result lineage

The v2 output directory already contains completed results. The following commands are read-only until an explicit `report`/`compare` write is requested:

```powershell
.\.venv\Scripts\python.exe -m continuum_robot.phase5.review_v2 check
.\.venv\Scripts\python.exe -m unittest tests.test_phase6_review_v2 -v
.\.venv\Scripts\python.exe scripts/check_docs_links.py
```

The GUI validates the saved registration, metadata, metrics, predictions and selected model/scaler hashes before using them. Missing or changed artifacts are reported in the interface. The exact experiment protocols and model lineage are in [`review_v2.py`](../src/continuum_robot/phase5/review_v2.py), with per-experiment `registration.json`, `metadata.json` and `provenance.json` under `results/final/phase6_review_v2/` and `artifacts/phase6_review_v2/`.

## Rebuild experiments only when intentionally requested

The next command **trains models and creates a fresh v2 result set**. Do not run it to launch the GUI or inspect existing results. The runner refuses to overwrite existing v2 outputs; choose new `--output` and `--models` paths for a deliberate rerun.

```powershell
.\.venv\Scripts\python.exe -m continuum_robot.phase5.review_v2 run --seed 42 --epochs 300 --output results/final/phase6_review_v2_rerun --models artifacts/phase6_review_v2_rerun
.\.venv\Scripts\python.exe -m continuum_robot.phase5.review_v2 report --output results/final/phase6_review_v2_rerun --models artifacts/phase6_review_v2_rerun
.\.venv\Scripts\python.exe -m continuum_robot.phase5.review_v2 compare --output results/final/phase6_review_v2_rerun --models artifacts/phase6_review_v2_rerun
```

Earlier synthetic pipeline commands and historical provenance are retained in the [archived technical README](legacy_readme.md). The scientific method is summarized in [methodology](methodology.md), [synthetic validation](synthetic_validation.md), and [physical validation](physical_validation.md).
