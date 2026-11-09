# Phase 6 review v2: physical test results

These results use original physical measurements, an assumed mm XYZ unit, and a PCC transform fitted on training measurements and selected on validation measurements. They do not identify the physical cable numbering or independently calibrate the sensor frame.

RQ1: Train-registered PCC RMSE = 4.970205 mm.
RQ2: FFNN RMSE = 0.388692 mm; compare against PCC on the same 4538 test rows.
RQ3: Linear 2.264401, Ridge2 1.530531, Ridge3 1.048315, KNN 0.320113, FFNN 0.388692 mm RMSE. A single static dataset and one split do not establish statistical significance.
RQ4: Sparse-grid test RMSE [mm]:

experiment method     N  RMSE_norm
  sparse_2    PCC 22625   4.950087
  sparse_2 Linear 22625   2.256444
  sparse_2 Ridge2 22625   1.497694
  sparse_2 Ridge3 22625   1.022305
  sparse_2    KNN 22625   0.407422
  sparse_2   FFNN 22625   0.537147
  sparse_5    PCC 29000   4.959545
  sparse_5 Linear 29000   2.263893
  sparse_5 Ridge2 29000   1.512295
  sparse_5 Ridge3 29000   1.037199
  sparse_5    KNN 29000   0.576929
  sparse_5   FFNN 29000   0.670995
 sparse_10    PCC 29925   4.971556
 sparse_10 Linear 29925   2.261710
 sparse_10 Ridge2 29925   1.519708
 sparse_10 Ridge3 29925   1.068483
 sparse_10    KNN 29925   0.863106
 sparse_10   FFNN 29925   0.964242

RQ5: Excluded-region test RMSE [mm]:

experiment method   N  RMSE_norm
  region_1    PCC 441   3.016191
  region_1 Linear 441   2.725760
  region_1 Ridge2 441   0.676540
  region_1 Ridge3 441   0.338909
  region_1    KNN 441   0.683911
  region_1   FFNN 441   0.227584
  region_2    PCC 441   4.233574
  region_2 Linear 441   1.677973
  region_2 Ridge2 441   0.981854
  region_2 Ridge3 441   0.839701
  region_2    KNN 441   0.655534
  region_2   FFNN 441   0.251214
  region_3    PCC 441   3.220858
  region_3 Linear 441   2.445439
  region_3 Ridge2 441   1.130799
  region_3 Ridge3 441   1.167773
  region_3    KNN 441   0.622387
  region_3   FFNN 441   0.323444

Limitations: no external coordinate/marker calibration, unknown measurement repeatability and history, one fixed split seed, no three-active-tendon observations, and no physical inverse-kinematics validation.
