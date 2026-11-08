# Coordinate conventions and empirical registration

The owner authorized an empirical, training-only registration when independent sensor calibration was unavailable. The reproducible pipeline is `python -m continuum_robot.phase5.empirical_registration`. For every evaluation protocol separately, it fits an orthogonal rotation/reflection and translation on that protocol's training rows; it chooses among six tendon permutations and both orientations using validation rows. Test XYZ values never participate in registration fitting or selection. The baseline is therefore **train-registered PCC**, and its error includes a data identification step. All twelve candidates, selected transform, row IDs and validation score are persisted under `data/real/experiments/<experiment>/prediction_provenance.json`. Dataset positions are assumed to be millimetres. No scale factor is fitted.

The external calibration route below remains available if sensor-frame metadata later arrives; it estimates a different baseline and should be reported separately.

The dataset contains physical measurements, but the attached material does not specify the measurement-frame origin, axis orientation, mapping of cables to the PCC indices, or the tracked marker geometry. The position unit is provisionally assumed to be mm. Command units are specified: one integer command unit = 0.1 mm tendon shortening.

The existing model has backbone length 110 mm and routing parameter 5 mm. It returns `[0, 0, 110]` at zero actuation. The physical dataset has no zero-actuation row. Its nearest rows are:

| Source row_id | Commands | Measured XYZ (source units) |
|---|---|---|
| 0 | (0,0,1) | (-43.06682,-144.2242,276.5916) |
| 100 | (0,1,0) | (-43.67575,-144.2628,276.7679) |
| 10200 | (1,0,0) | (-42.74495,-144.2724,277.2029) |

Direct subtraction of PCC-frame XYZ from measurement-frame XYZ gives an unaligned coordinate distance RMSE of 230.153540, which is **not a physical model error**. A global rigid fit with the nominal cable order gives 19.727866; three alternative cable permutations give 4.958635. These diagnostics assume millimetres and fit the entire dataset. They neither identify a unique cable order nor provide an independent calibration. Residual compensation must not be evaluated using a calibration fitted on held-out observations.

## Required information

Confirm the measured position units, physical backbone length and tendon routing radius, the dataset cable numbering relative to PCC, and whether XYZ locates the tip itself. Supply an independently established transform:

`p_measurement_mm = R @ p_PCC_mm + t_mm`

`R` is a 3x3 orthogonal axis transformation; `t` is a 3-vector translation. Coordinate systems may have different handedness, so an explicitly documented reflection is accepted. Three non-collinear independently measured corresponding reference points can determine a rigid registration; additional points allow checking its uncertainty. A single straight-pose measurement determines neither all axes nor cable numbering.

Copy `calibration.example.json` to `calibration.json` and fill the verified values and provenance. The null transform is intentional. An identity transform or fitted diagnostic must not be substituted solely to pass the gate.

The empirical registration is now implemented with the above protocol. Its fitted transform may absorb some model error, so comparisons to the original unregistered PCC must be interpreted carefully.

If the actual physical geometry differs from 110/5 mm or the sensor tracks an offset marker, the PCC integration needs those parameters or a pose-dependent marker correction before experiments can be scientifically interpreted.
