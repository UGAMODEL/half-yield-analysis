"""OpenCV operator display; measurement and saved video stay independent of the UI."""
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from calibration import Calibration, CalibrationControls
from hmi_style import HMIText, UGAStyle


class OperatorDashboard:
    WIDTH = 1280
    HEIGHT = 850
    BACKGROUND = UGAStyle.bgr(UGAStyle.SURFACE)
    INK = UGAStyle.bgr(UGAStyle.BLACK)
    MUTED = UGAStyle.bgr(UGAStyle.SANFORD)
    RED = UGAStyle.bgr(UGAStyle.RED)
    WHITE = UGAStyle.bgr(UGAStyle.WHITE)
    ACCENT = UGAStyle.bgr(UGAStyle.OLYMPIC)
    LINE = UGAStyle.bgr(UGAStyle.STEGEMAN)
    CONTROL = UGAStyle.bgr(UGAStyle.CONTROL)

    def __init__(self, window_name, source, window_seconds, skip_seconds):
        self.window_name = window_name
        self.source = Path(source).name
        self.window_seconds = window_seconds
        self.skip_seconds = skip_seconds
        self.calibration = Calibration()
        self.calibration_controls = CalibrationControls(self.calibration)
        self.buttons = ()
        self.actions = deque()
        self.pressed_action = None
        self.notice = 'CSV report is saved when you finish. Measurements use detected area, not weight.'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def render(self, frame, ratio, series, position, duration, paused, slow, complete=False):
        height = 1090 if self.calibration_controls.open else self.HEIGHT
        canvas = np.full((height, self.WIDTH, 3), self.BACKGROUND, dtype=np.uint8)
        self._render_buttons = []
        cv2.rectangle(canvas, (0, 0), (self.WIDTH, 87), self.RED, -1)
        cv2.rectangle(canvas, (0, 88), (self.WIDTH, 93), self.WHITE, -1)
        self._text(canvas, 'HALFYIELD  /  PROCESSING REVIEW', 28, 34, .6, self.WHITE, heading=True)
        filename = HMIText.fit(self.source, 700, 20)
        self._text(canvas, filename, 28, 67, .56, self.WHITE)
        state = 'REVIEW COMPLETE' if complete else ('PAUSED' if paused else 'ANALYZING')
        self._text(canvas, state, 970, 38, .6, self.WHITE, heading=True)
        self._text(canvas, 'Recorded video', 970, 67, .46, self.WHITE)
        self._button(canvas, (760, 22, 180, 42), 'Calibration', 'calibration')
        cv2.rectangle(canvas, (896, 100), (1252, 640), self.WHITE, -1)
        cv2.rectangle(canvas, (0, 648), (self.WIDTH, 766), self.WHITE, -1)
        self._video(canvas, frame)
        self._metrics(canvas, ratio, series)
        self._transport(canvas, position, duration, paused, slow, complete)
        self._footer(canvas)
        if self.calibration_controls.open:
            cv2.rectangle(canvas, (0, 850), (self.WIDTH, 1090), self.WHITE, -1)
            cv2.line(canvas, (28, 850), (1252, 850), self.LINE, 1)
            self.calibration_controls.draw(canvas, self)
        self.buttons = tuple(self._render_buttons)
        return canvas

    def read_action(self, key_action):
        action = self.actions.popleft() if self.actions else key_action
        if action == 'calibration':
            self.calibration_controls.open = not self.calibration_controls.open
            height = 1090 if self.calibration_controls.open else self.HEIGHT
            cv2.resizeWindow(self.window_name, self.WIDTH, height)
            return None
        return action

    def _footer(self, canvas):
        lines = (self.notice, 'Space: pause / resume     Arrows: seek or step     S: slow review     Q: finish')
        size, line_spacing = 16, 30
        bounds = [HMIText.bounds(line, size) for line in lines]
        top = min(box[1] + index * line_spacing for index, box in enumerate(bounds))
        bottom = max(box[3] + index * line_spacing for index, box in enumerate(bounds))
        baseline = round((767 + self.HEIGHT - (bottom - top)) / 2 - top)
        for index, line in enumerate(lines):
            self._text(canvas, line, 28, baseline + index * line_spacing, size / 36, self.MUTED)

    def _video(self, canvas, frame):
        x, y, w, h = 28, 100, 850, 510
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.INK, -1)
        scale = min(w / frame.shape[1], h / frame.shape[0])
        image = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                   max(1, round(frame.shape[0] * scale))))
        left = x + (w - image.shape[1]) // 2
        top = y + (h - image.shape[0]) // 2
        canvas[top:top + image.shape[0], left:left + image.shape[1]] = image
        self._text(canvas, 'INSPECTION VIEW', 28, 637, 0.46, self.MUTED)
        self._text(canvas, 'Outlined region is the fixed measurement area.', 210, 637, 0.46, self.MUTED)

    def _metrics(self, canvas, ratio, series):
        x = 910
        gap = 60
        self._text(canvas, 'HALF SHARE / AREA', x, 128, .6, self.RED, heading=True)
        value = '--' if ratio is None else f'{ratio * 100:.1f}%'
        self._text(canvas, value, x, 196, 1.85, self.INK, heading=True)
        self._text(canvas, f'Rolling {self.window_seconds:g}-second window', x, 224, 0.48, self.MUTED)
        self._text(canvas, 'Piece share', x, 224 + gap, 0.55)
        piece = '--' if ratio is None else f'{(1 - ratio) * 100:.1f}%'
        self._text(canvas, piece, x + 205, 224 + gap, 0.65)
        self._text(canvas, 'HALF SHARE TREND', x, 280 + gap, .55, self.INK, heading=True)
        self._chart(canvas, series, x, 300 + gap, 330, 130)
        y = 478 + gap
        for line in ('Half area / (half + piece area)', 'Shell and obscured excluded.', self._mass_label(ratio),
                     'Waiting for samples...' if ratio is None else 'Review alongside your plant QC checks.'):
            self._text(canvas, line, x, y, 0.45, self.MUTED)
            y += 25

    def _mass_label(self, ratio):
        share = self.calibration.mass_share(ratio)
        return 'Est. half mass share: --' if share is None else f'Est. half mass share: {share:.1%}'

    def _chart(self, canvas, series, x, y, width, height):
        for fraction in (0, 0.5, 1):
            row = y + round((1 - fraction) * height)
            cv2.line(canvas, (x, row), (x + width - 42, row), self.LINE, 1)
            self._text(canvas, f'{fraction * 100:.0f}%', x + width - 36, row + 5, 0.35, self.MUTED)
        valid = [(t, v) for t, v in series if v is not None and np.isfinite(v)]
        if len(valid) < 2:
            return
        start, end = valid[0][0], valid[-1][0]
        if end <= start:
            return
        points = np.array([(x + round((t - start) / (end - start) * (width - 42)),
                            y + round((1 - np.clip(v, 0, 1)) * height)) for t, v in valid], np.int32)
        cv2.polylines(canvas, [points], False, self.ACCENT, 2, cv2.LINE_AA)

    def _transport(self, canvas, position, duration, paused, slow, complete):
        progress = min(1, max(0, position / duration)) if duration > 0 else 0
        cv2.rectangle(canvas, (28, 659), (1252, 665), self.LINE, -1)
        cv2.rectangle(canvas, (28, 659), (28 + round(1224 * progress), 665), self.ACCENT, -1)
        self._text(canvas, f'{self._time(position)} / {self._time(duration)}', 28, 694, 0.48, self.MUTED)
        self._text(canvas, 'Seeking starts a new report segment.', 760, 694, 0.48, self.MUTED)
        play = 'Complete' if complete else ('Resume' if paused else 'Pause')
        self._button(canvas, (28, 714, 150, 44), play, None if complete else 'toggle_pause', True)
        self._button(canvas, (190, 714, 164, 44), f'Back {self.skip_seconds:g}s', 'seek_back')
        self._button(canvas, (366, 714, 164, 44), f'Ahead {self.skip_seconds:g}s', 'seek_forward')
        self._button(canvas, (542, 714, 168, 44), 'Previous frame', 'prev_frame')
        self._button(canvas, (722, 714, 150, 44), 'Next frame', 'next_frame')
        self._button(canvas, (884, 714, 172, 44), 'Slow: on' if slow else 'Slow: off', 'toggle_slow')
        self._button(canvas, (1068, 714, 184, 44), 'Finish / save CSV', 'quit', True)

    def _button(self, canvas, rect, label, action, primary=False):
        x, y, w, h = rect
        color = self.CONTROL
        foreground = self.INK
        if action == 'quit':
            color, foreground = self.RED, self.WHITE
        elif primary and action is not None:
            color, foreground = self.ACCENT, self.WHITE
        elif action == 'calibration':
            color, foreground = self.WHITE, self.RED
        elif action == 'toggle_slow' and label.endswith('on'):
            color, foreground = self.ACCENT, self.WHITE
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        text_width = HMIText.width(label, 16)
        self._text(canvas, label, x + (w - text_width) // 2, y + h // 2 + 6,
                   16 / 36, foreground)
        if action is not None and action == self.pressed_action:
            outline = self.WHITE if foreground == self.WHITE else self.INK
            cv2.rectangle(canvas, (x + 2, y + 2), (x + w - 2, y + h - 2), outline, 2)
        self._render_buttons.append((rect, action))

    def _on_mouse(self, event, x, y, flags, param):
        if self.calibration_controls.on_mouse(event, x, y):
            self.actions.append('calibration_changed')
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.pressed_action = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (left, top, width, height), action in self.buttons:
            if action is not None and left <= x < left + width and top <= y < top + height:
                self.pressed_action = action
                self.actions.append(action)
                break

    def _text(self, canvas, text, x, y, size=0.5, color=None, heading=False):
        pixels = max(14, round(size * 36))
        HMIText.draw(canvas, text, x, y, pixels, self.INK if color is None else color, heading)

    @staticmethod
    def _time(seconds):
        minutes, seconds = divmod(max(0, int(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}'
