"""Phase 6 verification (legacy module name phase5), using synthetic fixtures."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from continuum_robot.phase5 import pipeline as p
from continuum_robot.phase5 import empirical_registration as empirical


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        temporary_root = p.ROOT / "tmp" / "phase5_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.results = self.root / "results"
        self.models = self.root / "models"
        self.data.mkdir()
        self.overrides = patch.multiple(p, DATA=self.data, RESULTS=self.results, MODELS=self.models)
        self.overrides.start()
        self.cal = {"verified": True, "provenance": "Synthetic test fixture only; known coordinate transform",
                    "measurement_units_to_mm": 1., "dataset_columns_in_pcc_tendon_order": [2, 0, 1],
                    "backbone_length_mm": 110., "routing_radius_mm": 5., "position_marker": "tip",
                    "pcc_to_measurement_rotation": [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                    "pcc_to_measurement_translation_mm": [20., -40., 200.]}
        self.cal_path = self.data / "calibration.json"
        p.write_json(self.cal_path, self.cal)
        values = [0, 1, 2, 4, 5, 10, 20, 40, 50, 60, 80, 90, 100]
        commands = set()
        for a, b in itertools.product(values, repeat=2):
            commands.update(((a, b, 0), (a, 0, b), (0, a, b)))
        commands = np.array(sorted(commands), float)
        xyz = p.register_pcc(commands, self.cal)
        # Known smooth, nonzero residual, to exercise the target sign.
        xyz += commands / 100 + np.sin(commands / 20) * .05
        self.source = self.data / "fixture.txt"
        pd.DataFrame(np.column_stack((commands, xyz))).to_csv(self.source, index=False, header=False)

    def tearDown(self):
        self.overrides.stop()
        self.temp.cleanup()

    def test_calibration_gate_and_transform(self):
        with self.assertRaises(ValueError): p.read_calibration(None)
        invalid = dict(self.cal, verified=False)
        p.write_json(self.cal_path, invalid)
        with self.assertRaises(ValueError): p.read_calibration(self.cal_path)
        p.write_json(self.cal_path, self.cal)
        np.testing.assert_allclose(p.register_pcc(np.zeros((1, 3)), p.read_calibration(self.cal_path)), [[20, -40, 310]])
        invalid = dict(self.cal, pcc_to_measurement_rotation=np.diag([2, 1, 1]).tolist())
        p.write_json(self.cal_path, invalid)
        with self.assertRaises(ValueError): p.read_calibration(self.cal_path)

    def test_empirical_registration_excludes_test_positions(self):
        p.analyze(self.source, self.data)
        p.prepare(self.data, seed=42)
        split = json.loads((self.results / "splits" / "random.json").read_text())
        first = empirical.register_experiment("random", self.data)
        before = json.loads((first / "prediction_provenance.json").read_text())
        flagged = pd.read_csv(self.data / "real_dataset_flagged.csv")
        flagged.loc[flagged.row_id.isin(split["test"]), p.XYZ] += 1000
        flagged.to_csv(self.data / "real_dataset_flagged.csv", index=False)
        split["dataset_sha256"] = p.digest(self.data / "real_dataset_flagged.csv")
        p.write_json(self.results / "splits" / "random.json", split)
        second = empirical.register_experiment("random", self.data)
        after = json.loads((second / "prediction_provenance.json").read_text())
        self.assertEqual(before["tendon_order_selected"], after["tendon_order_selected"])
        np.testing.assert_allclose(before["row_rotation"], after["row_rotation"])
        np.testing.assert_allclose(before["translation_mm"], after["translation_mm"])
        self.assertEqual(before["selected_validation_RMSE_mm"], after["selected_validation_RMSE_mm"])

    def test_analysis_nonfinite_missing_flags_and_source_immutable(self):
        source_frame = pd.read_csv(self.source, header=None)
        # A spike at a configuration with immediate neighbours must be marked.
        source_frame.loc[(source_frame.iloc[:, :3] == [0, 1, 1]).all(axis=1), 3] += 15
        source_frame = pd.concat([source_frame, pd.DataFrame([[0, 0, 0, np.inf, 0, 0]])], ignore_index=True)
        source_frame.to_csv(self.source, header=False, index=False)
        before = p.digest(self.source)
        report = p.analyze(self.source, self.data)
        self.assertEqual(report["invalid_rows"], 1)
        self.assertGreater(report["flagged_rows"], 0)
        self.assertEqual(report["duplicate_configuration_extra_rows"], 1)
        self.assertEqual(before, p.digest(self.source))
        self.assertTrue((self.data / "missing_configurations.csv").exists())

    def test_full_pipeline_and_no_retraining_at_evaluation(self):
        p.analyze(self.source, self.data)
        frame = p.generate_predictions(self.cal_path, self.data)
        expected = frame[p.XYZ].to_numpy() - frame[p.PCC].to_numpy()
        np.testing.assert_allclose(frame[p.RESIDUAL], expected)
        p.prepare(self.data, seed=42)
        for name, train, val, test, spec in p.split_specs(frame, 42):
            split = json.loads((self.results / "splits" / f"{name}.json").read_text())
            self.assertEqual(len(set(train) & set(test)), 0)
            self.assertEqual(len(set(val) & set(test)), 0)
            if name.startswith("sparse"):
                step = int(name.split("_")[1])
                rows = frame.set_index("row_id").loc[train]
                self.assertTrue((rows[p.INPUTS].to_numpy() % step == 0).all())
            if name.startswith("region"):
                zero = spec["zero_tendon"] - 1
                active = [a for a in range(3) if a != zero]
                commands = frame.set_index("row_id").loc[test, p.INPUTS].to_numpy()
                self.assertTrue((commands[:, zero] == 0).all())
                self.assertTrue(((commands[:, active] >= 40) & (commands[:, active] <= 60)).all())
            p.train_experiment(name, self.data, epochs=3, seed=42)
            sx = joblib.load(self.models / name / "all" / "x_scaler.pkl")
            np.testing.assert_allclose(sx.mean_, frame.set_index("row_id").loc[train, p.INPUTS].to_numpy().mean(0)/10)
            artifacts = sorted((self.models / name).rglob("*"))
            hashes = {str(f): p.digest(f) for f in artifacts if f.is_file()}
            p.evaluate_experiment(name, self.data)
            self.assertEqual(hashes, {str(f): p.digest(f) for f in artifacts if f.is_file()})
            predictions = pd.read_csv(self.results / name / "predictions_all.csv")
            np.testing.assert_allclose(predictions.e_PCC, np.linalg.norm(predictions[p.PCC].to_numpy()-predictions[p.XYZ].to_numpy(), axis=1))
            self.assertEqual(set(predictions.row_id), set(test))
        for name in ("random", "sparse_2", "sparse_5", "sparse_10", "region_1", "region_2", "region_3"):
            p.plot_experiment(name)
        p.summarise()
        self.assertTrue((self.results / "research_answers.md").exists())
        self.assertTrue((self.results / "figures" / "sparse_grid_accuracy.png").exists())
        self.assertEqual(json.loads((self.results / "phase5_status.json").read_text())["status"], "complete")


if __name__ == "__main__": unittest.main()
