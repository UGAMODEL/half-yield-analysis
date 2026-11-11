"""Exercise browser endpoints and native-close replacement without model inference."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from halfyield.web_review import WebReview


class WebReviewTests(unittest.TestCase):
    def setUp(self):
        self.app = WebReview()
        self.base = f'http://127.0.0.1:{self.app.server.server_port}/{self.app.token}/'

    def tearDown(self):
        self.app.close()
        if self.app.worker:
            self.app.worker.join(timeout=5)
        self.app.server.shutdown()
        self.app.server.server_close()
        self.app.uploads.cleanup()

    def request(self, route, values=None, headers=None):
        body = json.dumps(values).encode() if values is not None else None
        request = Request(self.base + route, data=body, headers=headers or {})
        with urlopen(request, timeout=5) as response:
            return response.read()

    def test_html_and_assets_are_served_from_package(self):
        self.assertIn(b'id="start-form"', self.request(''))
        self.assertIn(b'pagehide', self.request('app.js'))
        self.assertIn(b'@font-face', self.request('style.css'))
        self.assertGreater(len(self.request('Oswald.ttf')), 1000)

    def test_close_beacon_wakes_runner(self):
        result = []
        thread = threading.Thread(target=lambda: result.append(self.app.show(np.zeros((32, 32, 3), np.uint8), 5000)))
        thread.start()
        self.request('quit', {})
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, ['quit'])

    def test_calibration_is_applied_on_runner_thread(self):
        values = dict(half_confidence=.1, piece_confidence=.3, shell_confidence=.4, obscured_confidence=.5,
                      half_mass_per_area_g_per_model_pixel2=.002, piece_mass_per_area_g_per_model_pixel2=.001)
        self.request('calibration', values)
        self.assertEqual(self.app.show(np.zeros((2, 2, 3), np.uint8), 0), 'calibration_changed')
        calibration = self.app.take_calibration()
        self.assertEqual(calibration.thresholds['half'], .1)
        self.assertEqual(calibration.half_mass_per_area, .002)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaises(HTTPError) as result:
            self.request('calibration', {})
        self.assertEqual(result.exception.code, 400)

    def test_unrelated_origin_cannot_control_runner(self):
        with self.assertRaises(HTTPError) as result:
            self.request('quit', {}, {'Origin': 'https://example.com'})
        self.assertEqual(result.exception.code, 403)
        self.assertFalse(self.app.closed)

    def test_browser_disconnect_exits_without_beacon(self):
        self.app.connected = True
        self.app.last_seen = time.monotonic() - 21
        self.assertEqual(self.app.show(np.zeros((2, 2, 3), np.uint8), 0), 'quit')

    def test_start_runs_inference_with_explicit_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'line.mp4'
            source.touch()
            report = Path(folder) / 'report.csv'
            def run(argv, review_window):
                self.assertIn(str(source.resolve()), argv)
                self.assertIs(review_window, self.app)
                report.write_text('report')
            with patch('halfyield.main.main', side_effect=run):
                self.request('start', {'source': str(source), 'report': str(report), 'window': 30})
                self.app.worker.join(timeout=5)
            state = json.loads(self.request('state'))
            self.assertEqual(state['status'], 'finished')
            self.assertEqual(state['report'], str(report.resolve()))

    def test_video_upload_keeps_original_filename(self):
        request = Request(self.base + 'upload', data=b'video-bytes',
                          headers={'X-Filename': 'line%20review.mp4'})
        with urlopen(request, timeout=5) as response:
            path = Path(json.loads(response.read())['source'])
        self.assertEqual(path.name, 'line review.mp4')
        self.assertEqual(path.read_bytes(), b'video-bytes')
        self.assertTrue(path.is_relative_to(self.app.uploads.name))

    def test_existing_report_cannot_be_overwritten_from_browser(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'line.mp4'
            source.touch()
            report = Path(folder) / 'report.csv'
            report.write_text('existing')
            with self.assertRaises(HTTPError):
                self.request('start', {'source': str(source), 'report': str(report)})
            self.assertEqual(report.read_text(), 'existing')
