"""Operator calibration values and their mouse-driven controls."""
from dataclasses import dataclass, field
import math

import cv2


@dataclass
class Calibration:
    thresholds: dict = field(default_factory=lambda: dict(half=.3, piece=.3, obscured=.3, shell=.3))
    half_mass_per_area: float = .001
    piece_mass_per_area: float = .001

    def mass_share(self, area_share):
        if area_share is None:
            return None
        half = area_share * self.half_mass_per_area
        piece = (1 - area_share) * self.piece_mass_per_area
        return half / (half + piece)

    def metadata(self):
        return {
            **{f'{name}_confidence': value for name, value in self.thresholds.items()},
            'half_mass_per_area_g_per_model_pixel2': self.half_mass_per_area,
            'piece_mass_per_area_g_per_model_pixel2': self.piece_mass_per_area,
        }


class CalibrationControls:
    FIELDS = (
        ('half', 'Half confidence', False),
        ('piece', 'Piece confidence', False),
        ('obscured', 'Obscured confidence', False),
        ('shell', 'Shell confidence', False),
        ('half_mass_per_area', 'Half mass / area', True),
        ('piece_mass_per_area', 'Piece mass / area', True),
    )

    def __init__(self, calibration):
        self.calibration = calibration
        self.open = False
        self.sliders = []
        self.dragging = None

    def draw(self, canvas, dashboard):
        self.sliders = []
        dashboard._text(canvas, 'CALIBRATION', 28, 872, .6, dashboard.RED, heading=True)
        dashboard._text(canvas, 'Changes start a new report segment. Mass values require weighed-sample calibration.',
                        210, 872, .45, dashboard.MUTED)
        for index, (name, label, mass) in enumerate(self.FIELDS):
            x = 28 + (index % 3) * 416
            y = 912 + (index // 3) * 84
            value = self._value(name, mass)
            displayed = f'{value:.6g} g/model px2' if mass else f'{value:.0%}'
            dashboard._text(canvas, label, x, y, .46)
            dashboard._text(canvas, displayed, x, y + 23, .44, dashboard.ACCENT)
            left, right, row = x, x + 364, y + 43
            cv2.line(canvas, (left, row), (right, row), dashboard.LINE, 4)
            fraction = (math.log10(value) + 6) / 6 if mass else value
            knob = round(left + max(0, min(1, fraction)) * (right - left))
            cv2.line(canvas, (left, row), (knob, row), dashboard.ACCENT, 4)
            cv2.circle(canvas, (knob, row), 8, dashboard.ACCENT, -1)
            self.sliders.append((left, right, row, name, mass))
        dashboard._text(canvas, 'Confidence: 0-100%. Mass coefficient: 0.000001-1 g/model px2 (log scale). Defaults are placeholders.',
                        28, 1070, .43, dashboard.MUTED)

    def on_mouse(self, event, x, y):
        if not self.open:
            return False
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = next((slider for slider in self.sliders
                                  if slider[0] - 10 <= x <= slider[1] + 10 and abs(y - slider[2]) <= 16), None)
        if self.dragging is None:
            return False
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            left, right, _, name, mass = self.dragging
            fraction = max(0, min(1, (x - left) / (right - left)))
            value = 10 ** (-6 + 6 * fraction) if mass else round(fraction, 2)
            if mass:
                setattr(self.calibration, name, value)
            else:
                self.calibration.thresholds[name] = value
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = None
            return True
        return False

    def _value(self, name, mass):
        return getattr(self.calibration, name) if mass else self.calibration.thresholds[name]
