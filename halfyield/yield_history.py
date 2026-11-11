"""Rolling area measurements with immutable yield values for plotted samples."""
from collections import deque
import math


class YieldHistory:
    def __init__(self, window_ms):
        self.window_ms = window_ms
        self.samples = deque()
        self.points = deque()

    @property
    def current(self):
        return self.points[-1][1] if self.points else None

    def append(self, timestamp, half_area, piece_area):
        if not all(math.isfinite(value) for value in (timestamp, half_area, piece_area)):
            return False
        if self.samples and timestamp <= self.samples[-1][0]:
            return False
        self.samples.append((timestamp, half_area, piece_area))
        cutoff = timestamp - self.window_ms
        # Keep the segment that overlaps the left edge of the integration window.
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        ratio = integrated_ratio_last(list(self.samples), self.window_ms)
        self.points.append((timestamp, ratio))
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()
        return True

    def clear(self):
        self.samples.clear()
        self.points.clear()


def integrated_ratio_last(samples, window_ms):
    """Integrate area over the window using each interval's left sample."""
    if len(samples) < 2:
        return None
    cutoff = samples[-1][0] - window_ms
    half_integral = 0.0
    total_integral = 0.0
    for index in range(len(samples) - 1):
        timestamp, half, piece = samples[index]
        next_timestamp = samples[index + 1][0]
        elapsed = next_timestamp - max(timestamp, cutoff)
        if elapsed <= 0:
            continue
        half_integral += half * elapsed
        total_integral += (half + piece) * elapsed
    return half_integral / total_integral if total_integral > 0 else None
