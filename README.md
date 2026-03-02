# continuum-robot-kinematics-error-compensation

This repository focuses on machine learning–based error compensation of continuum robot kinematics using data-driven models.


---

## Structure

```bash

generate_dataset.py      – synthetic PCC vs REAL dataset  
pcc_model.py             – analytical PCC model  
real_model.py            – simulated real model  
ml_models.py             – classical regression models  
train_nn_phase3.py       – neural network training  
eval_phase3_final.py     – final evaluation  

dataset_out/             – datasets  
models/                  – regression models  
phase3_version_3/        – trained neural network  
phase3_final/            – evaluation results  

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

2. Train neural network

```bash
python train_nn_phase3.py
```

3. Run final evaluation

```bash
python eval_phase3_final.py
```


![GUI for continuum robot kinematic error compensation](pic_dp/ex1.jpg)
![Residual Error Vectors in XYZ (TEST)](phase3_final/error_scatter3d_test.png)
![Error Norm Histogram (TEST)](phase3_final/errors_hist_test.png)
