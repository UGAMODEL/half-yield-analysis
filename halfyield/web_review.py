"""Loopback browser application. Inference stays in the Python runner."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import secrets
import tempfile
import threading
import time
import webbrowser

import cv2

from .calibration import Calibration
from .web_server import ReviewServer


class WebReview:
    def __init__(self):
        self.closed = False
        self.state = {'status': 'ready'}
        self.frame = b''
        self.actions = deque()
        self.condition = threading.Condition()
        self.pending_calibration = None
        self.worker = None
        self.connected = False
        self.last_seen = time.monotonic()
        self.token = secrets.token_urlsafe(24)
        self.uploads = tempfile.TemporaryDirectory(prefix='halfyield-')
        self.report = None
        self.server = ReviewServer(self)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def open(self):
        url = f'http://127.0.0.1:{self.server.server_port}/{self.token}/'
        print(f'HalfYield review: {url}', flush=True)
        webbrowser.open(url)

    def update(self, state):
        with self.condition:
            self.state.update(state, status='reviewing')

    def show(self, frame, delay):
        height, width = frame.shape[:2]
        if width > 1280:
            frame = cv2.resize(frame, (1280, round(height * 1280 / width)))
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.condition:
            if ok:
                self.frame = encoded.tobytes()
            if not self.actions and not self.closed:
                self.condition.wait(delay / 1000)
            # A heartbeat also handles browsers that discard the close beacon.
            timeout = 20 if self.connected else 120
            if time.monotonic() - self.last_seen > timeout:
                self.closed = True
            if self.closed:
                return 'quit'
            event = self.actions.popleft() if self.actions else None
            if isinstance(event, Calibration):
                self.pending_calibration = event
                return 'calibration_changed'
            return event

    def take_calibration(self):
        value, self.pending_calibration = self.pending_calibration, None
        return value

    def action(self, action):
        allowed = {'quit', 'toggle_pause', 'toggle_slow', 'seek_back', 'seek_forward',
                   'prev_frame', 'next_frame', 'left', 'right'}
        if action not in allowed:
            raise ValueError('Unknown review action')
        with self.condition:
            if action == 'quit':
                self.closed = True
            else:
                self.actions.append(action)
            self.condition.notify_all()

    def calibrate(self, values):
        thresholds = {name: float(values[name + '_confidence'])
                      for name in ('half', 'piece', 'shell', 'obscured')}
        half = float(values['half_mass_per_area_g_per_model_pixel2'])
        piece = float(values['piece_mass_per_area_g_per_model_pixel2'])
        if not all(0 <= value <= 1 for value in thresholds.values()):
            raise ValueError('Confidence must be between zero and one')
        if not all(1e-6 <= value <= 1 for value in (half, piece)):
            raise ValueError('Mass coefficients must be between 0.000001 and 1')
        with self.condition:
            self.actions.append(Calibration(thresholds, half, piece))
            self.condition.notify_all()

    def start(self, values):
        with self.condition:
            if self.worker is not None:
                raise ValueError('A review has already started')
            source = Path(values['source']).expanduser().resolve()
            if not source.is_file():
                raise ValueError('Choose an existing video')
            report_text = values.get('report', '').strip()
            if report_text:
                report = Path(report_text).expanduser().resolve()
            else:
                stamp = time.strftime('%Y%m%d_%H%M%S')
                report = Path.cwd() / f'{source.stem}_review_{stamp}.csv'
            if report == source or report.suffix.lower() != '.csv' or not report.parent.is_dir():
                raise ValueError('Choose a CSV file in an existing folder')
            if report.exists():
                raise ValueError('The report already exists. Choose a new filename.')
            window = float(values.get('window', 60))
            if not 0 < window <= 86400:
                raise ValueError('Choose a positive measurement window up to 86400 seconds')
            self.report = report
            self.state = {'status': 'loading', 'source': source.name}
            self.worker = threading.Thread(target=self._run, args=(source, report, window), daemon=True)
            self.worker.start()

    def _run(self, source, report, window):
        try:
            from .main import main
            main(['--source', str(source), '--show', '--yield-output', str(report),
                  '--ratio-window-sec', str(window)], review_window=self)
        except (Exception, SystemExit) as error:
            with self.condition:
                self.state.update(status='error', error=str(error))
        finally:
            self.close()

    def close(self):
        with self.condition:
            self.closed = True
            if self.state.get('status') != 'error':
                self.state['status'] = 'finished'
            if self.report and self.report.is_file():
                self.state['report'] = str(self.report)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            self.connected = True
            self.last_seen = time.monotonic()
            return deepcopy(self.state)

    def wait(self):
        try:
            while True:
                with self.condition:
                    if self.closed and (self.worker is None or not self.worker.is_alive()):
                        break
                    if self.connected and time.monotonic() - self.last_seen > 20:
                        self.closed = True
                        self.condition.notify_all()
                time.sleep(.1)
        except KeyboardInterrupt:
            self.action('quit')
        finally:
            if self.worker:
                self.worker.join()
            # Let the browser receive final report status before stopping HTTP.
            time.sleep(1)
            self.server.shutdown()
            self.server.server_close()
            self.uploads.cleanup()


def launch():
    app = WebReview()
    app.open()
    app.wait()
