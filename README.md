# continuum-robot-kinematics-error-compensation

This repository focuses on machine learning–based error compensation of continuum robot kinematics using data-driven models.


---

## Structure

```bash

generate_dataset.py      – synthetic PCC vs REAL dataset  
csv_to_nzp.py            – converts canonical CSV dataset to NPZ
pcc_model.py             – analytical PCC model  
real_model.py            – simulated real model  
ml_models.py             – classical regression models  
train_nn_phase3.py       – neural network training  
eval_phase3_final.py     – final evaluation  

dataset_out/             – datasets
  dataset_3.csv          – canonical Phase 2/3 CSV dataset
  dataset_3.npz          – canonical Phase 2/3 NPZ dataset
models/                  – legacy classical regression models used by GUI
phase3_last_16_03_2026/  – canonical Phase 3 neural network output
phase3_final/            – evaluation results  
phase3_version_3/        – legacy neural-network artifact

```

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn torch joblib
```

---

## Workflow

1. Generate dataset

```bash
python generate_dataset.py
```

2. Convert the canonical CSV dataset to NPZ

```bash
python csv_to_nzp.py
```

3. Train neural network

```bash
python train_nn_phase3.py
```

4. Run final evaluation

```bash
python eval_phase3_final.py
```

Canonical Phase 2/3 paths:

```bash
dataset_out/dataset_3.csv
dataset_out/dataset_3.npz
phase3_last_16_03_2026/
phase3_final/
```

Legacy artifacts such as `dataset_phase2_synth_new.npz`, `phase3_version_3/`, and saved sklearn models in `models/` are kept for reference and GUI compatibility, but they are not the canonical Phase 2/3 pipeline.


![GUI for continuum robot kinematic error compensation](pic_dp/ex1.jpg)
![Residual Error Vectors in XYZ (TEST)](phase3_final/error_scatter3d_test.png)
![Error Norm Histogram (TEST)](phase3_final/errors_hist_test.png)
