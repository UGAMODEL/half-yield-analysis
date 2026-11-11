"""Exercise playback transitions without a display or model inference."""
from collections import deque
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main
from operator_dashboard import OperatorDashboard


class VideoCapture:
    def __init__(self):
        self.position = 0
        self.total = 5

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_COUNT: self.total,
            cv2.CAP_PROP_FPS: 1,
            cv2.CAP_PROP_FRAME_WIDTH: 32,
            cv2.CAP_PROP_FRAME_HEIGHT: 32,
            cv2.CAP_PROP_POS_FRAMES: self.position,
            cv2.CAP_PROP_POS_MSEC: max(0, self.position - 1) * 1000,
        }.get(prop, 0)

    def read(self):
        if self.position >= self.total:
            return False, None
        self.position += 1
        return True, np.zeros((32, 32, 3), np.uint8)

    def set(self, prop, value):
        self.position = int(value if prop == cv2.CAP_PROP_POS_FRAMES else value / 1000)

    def release(self):
        pass


class OperatorWorkflowTests(unittest.TestCase):
    def run_review(self, keys, extra=(), calibration_at=None):
        capture = VideoCapture()
        model = unittest.mock.Mock(names={0: 'half', 1: 'piece'})
        samples = []
        self.inference_settings = []
        read_action = OperatorDashboard.read_action
        action_count = 0
        callback = None
        events = iter(keys)
        self.playback_states = []
        original_render = OperatorDashboard.render
        def render(dashboard, frame, ratio, series, position, duration, paused, slow, complete=False):
            self.playback_states.append(paused)
            return original_render(dashboard, frame, ratio, series, position, duration, paused, slow, complete)
        def set_mouse_callback(window, handler):
            nonlocal callback
            callback = handler
        def wait_key(delay):
            event = next(events)
            if isinstance(event, tuple):
                kind, x, y = event
                callback(kind, x, y, 0, None)
                return -1
            return event
        def calibrate(dashboard, key):
            nonlocal action_count
            action_count += 1
            if action_count == calibration_at:
                dashboard.calibration.thresholds['half'] = .1
                dashboard.calibration.half_mass_per_area = .002
                return 'calibration_changed'
            return read_action(dashboard, key)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'line.mp4'
            source.touch()
            def process(frame, *args, **kwargs):
                self.inference_settings.append((kwargs['base_conf'], kwargs['class_thresh'].copy()))
                return frame, 75, 25
            def save(stats, output, video):
                samples.append(stats)
            with patch.multiple(main, YOLO=unittest.mock.Mock(return_value=model),
                                get_optimal_device=lambda: 'cpu',
                                create_video_capture=lambda *a, **k: capture,
                                process_frame=process,
                                save_yield_report=save), \
                 patch.multiple(cv2, namedWindow=lambda *a: None, resizeWindow=lambda *a: None,
                                setMouseCallback=set_mouse_callback, imshow=lambda *a: None,
                                waitKey=wait_key,
                                getWindowProperty=lambda *a: 1, destroyAllWindows=lambda: None), \
                 patch.object(OperatorDashboard, 'read_action', calibrate), \
                 patch.object(OperatorDashboard, 'render', render), \
                 patch('sys.argv', ['main.py', '--source', str(source), '--show', *extra]):
                main.main()
        return samples

    def test_pause_button_uses_same_playback_action_as_space(self):
        down = (cv2.EVENT_LBUTTONDOWN, 100, 736)
        up = (cv2.EVENT_LBUTTONUP, 100, 736)
        self.run_review([down, up, down, up, ord('q')])
        self.assertEqual(self.playback_states, [False, True, True, False, False])
        self.run_review([ord(' '), -1, ord(' '), -1, ord('q')])
        self.assertEqual(self.playback_states, [False, True, True, False, False])

    def test_calibration_change_resets_history_and_records_new_settings(self):
        reports = self.run_review([-1] * 6 + [ord('q')], calibration_at=3)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['duration_sec'], 1)
        self.assertEqual(reports[0]['calibration']['half_confidence'], .1)
        self.assertEqual(reports[0]['calibration']['half_mass_per_area_g_per_model_pixel2'], .002)
        self.assertAlmostEqual(reports[0]['estimated_half_mass_share'], 6 / 7)
        self.assertEqual(self.inference_settings[-1][0], .1)

    def test_class_threshold_below_base_lowers_inference_prefilter(self):
        self.run_review([-1, ord('q')], ('--half-conf', '0.1'))
        base, thresholds = self.inference_settings[0]
        self.assertEqual(base, .1)
        self.assertEqual(thresholds['half'], .1)
        self.assertEqual(thresholds['piece'], .3)

    def test_end_of_video_retains_report_until_finish(self):
        reports = self.run_review([-1] * 6 + [ord('q')])
        self.assertEqual(len(reports), 1)
        self.assertAlmostEqual(reports[0]['net_average'], .75)
        self.assertEqual(reports[0]['duration_sec'], 4)

    def test_backward_seek_removes_previous_report_segment(self):
        reports = self.run_review([-1, -1, ord(','), ord(' '), -1, -1, ord('q')])
        self.assertEqual(len(reports), 1)
        self.assertGreater(reports[0]['duration_sec'], 0)
        self.assertAlmostEqual(reports[0]['net_average'], .75)

    def test_frame_step_accounts_for_next_unread_position(self):
        capture = VideoCapture()
        capture.read()
        capture.read()
        main.apply_step_frame(capture, -2)
        capture.read()
        self.assertEqual(capture.position, 1)

    def test_empty_measurement_is_not_zero_percent(self):
        self.assertIsNone(main.integrated_ratio_last(deque([(0, 0, 0), (1000, 0, 0)]), 60000))


if __name__ == '__main__':
    unittest.main()
