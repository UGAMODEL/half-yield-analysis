"""Exercise button callbacks, including clicks that span multiple display frames."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from halfyield.main import parse_key
from halfyield.operator_dashboard import OperatorDashboard


class DashboardButtonTests(unittest.TestCase):
    def setUp(self):
        with patch.multiple(cv2, namedWindow=lambda *a: None, resizeWindow=lambda *a: None,
                            setMouseCallback=lambda *a: None):
            self.dashboard = OperatorDashboard('test', 'line.mp4', 60, 5)
        self.frame = np.zeros((32, 32, 3), np.uint8)
        self.render()

    def render(self, complete=False):
        return self.dashboard.render(self.frame, .75, [], 1, 5, False, False, complete)

    def test_playback_buttons_match_keyboard_actions(self):
        for x, key in [(100, ' '), (610, ','), (790, '.'), (960, 's'), (1150, 'q')]:
            with self.subTest(key=key):
                self.dashboard._on_mouse(cv2.EVENT_LBUTTONDOWN, x, 736, 0, None)
                self.assertEqual(self.dashboard.read_action(None), parse_key(ord(key)))
                self.render()
                self.dashboard._on_mouse(cv2.EVENT_LBUTTONUP, x, 736, 0, None)
                self.assertIsNone(self.dashboard.read_action(None))

    def test_quick_clicks_are_queued_in_order(self):
        self.dashboard._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 736, 0, None)
        self.dashboard._on_mouse(cv2.EVENT_LBUTTONUP, 100, 736, 0, None)
        self.dashboard._on_mouse(cv2.EVENT_LBUTTONDOWN, 960, 736, 0, None)
        self.assertEqual(self.dashboard.read_action(None), 'toggle_pause')
        self.assertEqual(self.dashboard.read_action(None), 'toggle_slow')
        self.assertIsNone(self.dashboard.read_action(None))

    def test_completed_playback_button_is_disabled(self):
        self.render(complete=True)
        self.dashboard._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 736, 0, None)
        self.assertIsNone(self.dashboard.read_action(None))


if __name__ == '__main__':
    unittest.main()
