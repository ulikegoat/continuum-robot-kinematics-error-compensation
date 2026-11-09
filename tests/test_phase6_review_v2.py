"""Fast, synthetic-only checks for the new physical-data protocol."""
import unittest

import numpy as np

from continuum_robot.phase5.review_v2 import choose_registration, rigid_fit


class RegistrationProtocolTests(unittest.TestCase):
    def test_rigid_fit_recovers_known_mapping_without_scale(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 2, 3], [2, 1, 4]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = source @ rotation + [10, -20, 30]
        fitted_rotation, translation = rigid_fit(source, target, 1)
        np.testing.assert_allclose(source @ fitted_rotation + translation, target, atol=1e-12)
        np.testing.assert_allclose(fitted_rotation.T @ fitted_rotation, np.eye(3), atol=1e-12)

    def test_test_positions_never_select_registration(self):
        rng = np.random.default_rng(19)
        commands = rng.uniform(0, 100, (30, 3))
        source = commands / 10
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        measured = source @ rotation + [10, -20, 30]
        train, validation = np.arange(20), np.arange(20, 25)
        first, selected, candidates = choose_registration(
            commands, measured, train, validation, pcc_forward=lambda x: x)
        changed = measured.copy()
        changed[25:] += 1000
        second, selected_after, candidates_after = choose_registration(
            commands, changed, train, validation, pcc_forward=lambda x: x)
        self.assertEqual(selected, selected_after)
        self.assertEqual(candidates, candidates_after)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first[:25], measured[:25], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
