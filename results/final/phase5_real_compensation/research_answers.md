# Phase 6 physical measurement results

PCC was rigidly registered from training data only; tendon permutation selected on validation data. Measurement positions assumed mm. This is a data-identified baseline, not independently measured calibration.
The six tendon permutations are geometrically symmetric after allowing arbitrary rigid rotations/reflections; their nearly identical validation scores do not identify the physical cable numbering. The selected permutation is an internal coordinate convention.
Primary results retain all valid measurements, including QA flags. Clean sensitivity results are conditional on globally computed descriptive flags.

RQ1: Registered PCC random-test RMSE = 4.970205; MAE = 4.666186; P95 = 7.840717.
RQ2: Best compensation (FFNN) RMSE = 0.388837, reduction = 92.18%.
RQ3: FFNN RMSE = 0.388837; Ridge3 RMSE = 1.048315. Numerical differences alone do not establish statistical significance.
RQ4: Sparse grid test RMSE:

experiment  model     N  RMSE_norm
 sparse_10    PCC 29925   4.971556
 sparse_10 Linear 29925   2.261710
 sparse_10 Ridge2 29925   1.519708
 sparse_10 Ridge3 29925   1.068483
 sparse_10   FFNN 29925   0.964242
  sparse_2    PCC 22625   4.950087
  sparse_2 Linear 22625   2.256444
  sparse_2 Ridge2 22625   1.497694
  sparse_2 Ridge3 22625   1.022305
  sparse_2   FFNN 22625   0.537147
  sparse_5    PCC 29000   4.959545
  sparse_5 Linear 29000   2.263893
  sparse_5 Ridge2 29000   1.512295
  sparse_5 Ridge3 29000   1.037199
  sparse_5   FFNN 29000   0.667351

RQ5: Contiguous region test RMSE (each plane):

experiment  model   N  RMSE_norm
  region_1    PCC 441   3.016191
  region_1 Linear 441   2.725760
  region_1 Ridge2 441   0.676540
  region_1 Ridge3 441   0.338909
  region_1   FFNN 441   0.162205
  region_2    PCC 441   4.233574
  region_2 Linear 441   1.677973
  region_2 Ridge2 441   0.981854
  region_2 Ridge3 441   0.839701
  region_2   FFNN 441   0.279543
  region_3    PCC 441   3.220858
  region_3 Linear 441   2.445439
  region_3 Ridge2 441   1.130799
  region_3 Ridge3 441   1.167773
  region_3   FFNN 441   0.276882

Limitations: static positions, at most two active tendons, single seed, unknown repeatability and motion history; no physical IK validation and no evidence for three-tendon configurations.