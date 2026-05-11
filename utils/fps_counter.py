# =============================================================================
# utils/fps_counter.py
# Rolling Average FPS Counter
# =============================================================================

import time
from collections import deque


class FPSCounter:
    """
    Menghitung FPS menggunakan rolling window average.

    Menggunakan deque dengan maxlen untuk menyimpan timestamp
    N frame terakhir. FPS = N / (t_newest - t_oldest).
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Jumlah frame dalam rolling window
        """
        self._timestamps = deque(maxlen=window_size)

    def tick(self):
        """Panggil sekali per frame, sebelum atau sesudah proses."""
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Kembalikan estimasi FPS saat ini."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed