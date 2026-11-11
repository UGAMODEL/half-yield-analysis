"""Regression checks for fixed chart values as raw integration samples expire."""
import unittest
from unittest.mock import patch

from yield_history import YieldHistory, integrated_ratio_last


class YieldHistoryTests(unittest.TestCase):
    def test_existing_points_do_not_change_when_raw_samples_expire(self):
        history = YieldHistory(2000)
        history.append(0, 100, 0)
        history.append(1000, 0, 100)
        history.append(2000, 100, 0)
        original = dict(history.points)
        self.assertEqual(original[1000], 1)
        self.assertEqual(original[2000], .5)

        history.append(3000, 0, 100)
        self.assertEqual(history.samples[0][0], 1000)
        for timestamp in (1000, 2000):
            self.assertEqual(dict(history.points)[timestamp], original[timestamp])

        history.append(4000, 100, 0)
        self.assertEqual(history.samples[0][0], 2000)
        self.assertEqual(dict(history.points)[2000], original[2000])
        self.assertNotIn(1000, dict(history.points))

    def test_ratio_is_calculated_once_per_accepted_point(self):
        history = YieldHistory(2000)
        with patch('yield_history.integrated_ratio_last', wraps=integrated_ratio_last) as calculate:
            history.append(0, 3, 1)
            history.append(1000, 3, 1)
            history.append(1000, 1, 3)
            for _ in range(3):
                self.assertEqual(history.current, .75)
                list(history.points)
            self.assertEqual(calculate.call_count, 2)

    def test_partial_interval_at_left_boundary_contributes_to_current_ratio(self):
        history = YieldHistory(2500)
        history.append(0, 2, 0)
        history.append(1000, 0, 2)
        history.append(3000, 2, 0)
        self.assertAlmostEqual(history.current, .2)
        self.assertEqual(history.samples[0][0], 0)

    def test_repeated_or_backward_timestamp_does_not_add_a_point(self):
        history = YieldHistory(2000)
        history.append(0, 3, 1)
        history.append(1000, 0, 4)
        original = list(history.points)
        self.assertFalse(history.append(1000, 4, 0))
        self.assertFalse(history.append(500, 4, 0))
        self.assertEqual(list(history.points), original)

    def test_empty_measurement_remains_a_gap(self):
        history = YieldHistory(1000)
        history.append(0, 0, 0)
        history.append(1000, 0, 0)
        self.assertIsNone(history.current)
        self.assertEqual(list(history.points), [(0, None), (1000, None)])

    def test_clear_removes_points_and_integration_samples(self):
        history = YieldHistory(2000)
        history.append(0, 3, 1)
        history.append(1000, 3, 1)
        history.clear()
        self.assertIsNone(history.current)
        self.assertFalse(history.samples)
        self.assertFalse(history.points)
        self.assertTrue(history.append(0, 1, 3))
        self.assertIsNone(history.current)

    def test_long_gap_uses_overlapping_sample_without_changing_old_values(self):
        history = YieldHistory(2000)
        history.append(0, 3, 1)
        history.append(10000, 1, 3)
        self.assertEqual(list(history.points), [(10000, .75)])


if __name__ == '__main__':
    unittest.main()
