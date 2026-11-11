"""Check calibration math, slider ranges, and exported measurement context."""
import csv
from pathlib import Path
import tempfile
import unittest

import cv2

from halfyield.calibration import Calibration, CalibrationControls
from halfyield.main import calculate_yield_statistics, save_yield_report


class CalibrationTests(unittest.TestCase):
    def test_separate_coefficients_change_mass_share(self):
        values = Calibration(half_mass_per_area=.002, piece_mass_per_area=.001)
        self.assertAlmostEqual(values.mass_share(.75), 6 / 7)
        self.assertIsNone(values.mass_share(None))
        self.assertEqual(values.mass_share(0), 0)
        self.assertEqual(values.mass_share(1), 1)

    def test_common_coefficient_scale_does_not_change_share(self):
        first = Calibration(half_mass_per_area=.002, piece_mass_per_area=.001)
        second = Calibration(half_mass_per_area=.02, piece_mass_per_area=.01)
        self.assertAlmostEqual(first.mass_share(.6), second.mass_share(.6))

    def test_sliders_apply_on_release_and_clamp_drag(self):
        controls = CalibrationControls(Calibration())
        controls.open = True
        controls.sliders = [(10, 110, 20, 'half', False)]
        self.assertFalse(controls.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 20))
        self.assertEqual(controls.calibration.thresholds['half'], .5)
        self.assertFalse(controls.on_mouse(cv2.EVENT_MOUSEMOVE, -100, 20))
        self.assertTrue(controls.on_mouse(cv2.EVENT_LBUTTONUP, -100, 20))
        self.assertEqual(controls.calibration.thresholds['half'], 0)
        self.assertIsNone(controls.dragging)

    def test_mass_slider_has_logarithmic_scale(self):
        controls = CalibrationControls(Calibration())
        controls.open = True
        controls.sliders = [(10, 110, 20, 'piece_mass_per_area', True)]
        controls.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 20)
        controls.on_mouse(cv2.EVENT_LBUTTONUP, 60, 20)
        self.assertAlmostEqual(controls.calibration.piece_mass_per_area, .001)

    def test_csv_records_coefficients_and_separate_mass_share(self):
        values = Calibration(half_mass_per_area=.002, piece_mass_per_area=.001)
        stats = calculate_yield_statistics([(0, 75, 25), (1000, 75, 25)])
        stats['calibration'] = values.metadata()
        stats['estimated_half_mass_share'] = values.mass_share(stats['net_average'])
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'report.csv'
            save_yield_report(stats, report, 'line.mp4')
            with report.open(newline='') as stream:
                rows = list(csv.reader(stream))
        self.assertIn(['# Calibration half_mass_per_area_g_per_model_pixel2:', '0.002'], rows)
        header_index = next(i for i, row in enumerate(rows) if row and row[0] == 'Time_Start_Seconds')
        self.assertEqual(rows[header_index][-1], 'Estimated_Half_Mass_Share')
        self.assertAlmostEqual(float(rows[header_index + 1][-1]), 6 / 7)
        self.assertAlmostEqual(float(rows[header_index + 1][1]), .75)


if __name__ == '__main__':
    unittest.main()
