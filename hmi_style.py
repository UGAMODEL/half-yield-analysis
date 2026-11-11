"""UGA visual-style tokens and cached font rendering for the operator HMI."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UGAStyle:
    RED = '#BA0C2F'
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    STEGEMAN = '#9EA2A2'
    SANFORD = '#554F47'
    OLYMPIC = '#004E60'
    SURFACE = '#E2E3E3'  # Stegeman at 30% on white.
    CONTROL = '#CFD0D0'  # Stegeman at 50% on white.

    @staticmethod
    def bgr(hex_color):
        return tuple(int(hex_color[index:index + 2], 16) for index in (5, 3, 1))


class HMIText:
    @staticmethod
    def draw(canvas, text, x, baseline, size, color, heading=False):
        mask, left, top = HMIText._mask(str(text), size, heading)
        x, y = x + left, baseline + top
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + mask.shape[1]), min(canvas.shape[0], y + mask.shape[0])
        if x0 >= x1 or y0 >= y1:
            return
        alpha = mask[y0 - y:y1 - y, x0 - x:x1 - x, None] / 255.0
        region = canvas[y0:y1, x0:x1]
        region[:] = (region * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    @staticmethod
    def width(text, size, heading=False):
        return round(HMIText._font(size, heading).getlength(text))

    @staticmethod
    def bounds(text, size, heading=False):
        return HMIText._font(size, heading).getbbox(text, anchor='ls')

    @staticmethod
    def fit(text, width, size, heading=False):
        if HMIText.width(text, size, heading) <= width:
            return text
        while text and HMIText.width(text + '…', size, heading) > width:
            text = text[:-1]
        return text + '…'

    @staticmethod
    @lru_cache(maxsize=48)
    def _font(size, heading):
        name = 'Oswald.ttf' if heading else 'MerriweatherSans.ttf'
        font = ImageFont.truetype(str(Path(__file__).parent / 'assets' / 'fonts' / name), size)
        font.set_variation_by_axes([500 if heading else 400])
        return font

    @staticmethod
    @lru_cache(maxsize=512)
    def _mask(text, size, heading):
        font = HMIText._font(size, heading)
        left, top, right, bottom = font.getbbox(text, anchor='ls')
        image = Image.new('L', (max(1, right - left), max(1, bottom - top)))
        ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=255, anchor='ls')
        return np.asarray(image), left, top


class InspectionOverlay:
    @staticmethod
    def draw(frame, polygons, names):
        import cv2

        white = UGAStyle.bgr(UGAStyle.WHITE)
        ink = UGAStyle.bgr(UGAStyle.BLACK)
        accent = UGAStyle.bgr(UGAStyle.OLYMPIC)
        for polygon, class_id in polygons:
            points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], True, white, 4, cv2.LINE_AA)
            cv2.polylines(frame, [points], True, accent, 2, cv2.LINE_AA)
            moments = cv2.moments(points)
            if moments['m00'] == 0:
                continue
            label = HMIText.fit(str(names.get(int(class_id), class_id)), max(1, frame.shape[1] - 16), 18)
            width = HMIText.width(label, 18) + 12
            x = min(max(0, int(moments['m10'] / moments['m00'])), max(0, frame.shape[1] - width))
            y = min(max(28, int(moments['m01'] / moments['m00']) - 10), frame.shape[0] - 1)
            cv2.rectangle(frame, (x, y - 26), (x + width, y), white, -1)
            HMIText.draw(frame, label, x + 6, y - 6, 18, ink)
