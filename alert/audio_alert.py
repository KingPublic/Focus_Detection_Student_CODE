# =============================================================================
# alert/audio_alert.py
# Audio Alert Manager
#
# Menggunakan sounddevice + numpy untuk generate beep tone secara programatik.
# Tidak memerlukan file audio eksternal (100% self-contained).
#
# Jika sounddevice tidak tersedia:
#   - Windows: fallback ke winsound.Beep
#   - Linux/Mac: fallback ke terminal bell (\a)
#
# Audio diputar di thread terpisah agar tidak memblokir main loop.
# =============================================================================

import time
import threading
import sys
from typing import Optional

# --- Coba import sounddevice ---
try:
    import numpy as np
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

# --- Fallback winsound ---
_HAS_WINSOUND = False
if sys.platform == "win32" and not _HAS_SOUNDDEVICE:
    try:
        import winsound
        _HAS_WINSOUND = True
    except ImportError:
        pass


class AudioAlert:
    """
    Manajemen audio alert dengan:
        - Dua level: WARNING (kuning) dan CRITICAL (merah)
        - Cooldown: mencegah spam beep berulang
        - Non-blocking: menggunakan background thread
        - Fade in/out: mencegah click artifacts
    """

    # Frekuensi beep (Hz)
    FREQ_WARNING  = 880    # A5 — peringatan
    FREQ_CRITICAL = 1400   # Lebih tinggi — kritis
    FREQ_CLEAR    = 440    # A4 — kondisi kembali normal

    SAMPLE_RATE   = 44100  # Standard audio sample rate

    def __init__(self):
        self._last_warn_time: float = 0.0
        self._last_crit_time: float = 0.0
        self._lock = threading.Lock()
        self._playing = False

        backend = "sounddevice" if _HAS_SOUNDDEVICE else \
                  ("winsound" if _HAS_WINSOUND else "terminal bell")
        print(f"[AudioAlert] Initialized — backend: {backend}")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def alert_warning(self, cooldown: float = 5.0):
        """
        Putar beep level WARNING.

        Args:
            cooldown: Interval minimum antar beep (detik)
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_warn_time < cooldown:
                return
            self._last_warn_time = now
        self._play_async(self.FREQ_WARNING, duration=0.35, volume=0.5)

    def alert_critical(self, cooldown: float = 3.0):
        """
        Putar beep level CRITICAL (lebih keras dan panjang).

        Args:
            cooldown: Interval minimum antar beep (detik)
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_crit_time < cooldown:
                return
            self._last_crit_time = now
        # Double-beep untuk critical
        self._play_async(self.FREQ_CRITICAL, duration=0.25, volume=0.7)
        time.sleep(0.08)
        self._play_async(self.FREQ_CRITICAL, duration=0.35, volume=0.7)

    def alert_clear(self):
        """Putar nada singkat saat sistem kembali ke kondisi OK."""
        self._play_async(self.FREQ_CLEAR, duration=0.15, volume=0.3)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _play_async(self, freq: float, duration: float, volume: float):
        """Jalankan pemutaran audio di background thread."""
        t = threading.Thread(
            target=self._play_tone,
            args=(freq, duration, volume),
            daemon=True,
        )
        t.start()

    def _play_tone(self, freq: float, duration: float, volume: float):
        """Putar satu nada sinus dengan fade in/out."""
        if _HAS_SOUNDDEVICE:
            self._play_sounddevice(freq, duration, volume)
        elif _HAS_WINSOUND:
            self._play_winsound(freq, duration)
        else:
            print("\a", end="", flush=True)  # terminal bell fallback

    def _play_sounddevice(self, freq: float, duration: float, volume: float):
        """Generate dan putar tone menggunakan sounddevice."""
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Gelombang sinus dasar
        wave = volume * np.sin(2 * np.pi * freq * t)

        # Fade in/out (10ms) untuk menghilangkan click
        fade_len = min(int(self.SAMPLE_RATE * 0.01), n_samples // 4)
        wave[:fade_len]  *= np.linspace(0, 1, fade_len)
        wave[-fade_len:] *= np.linspace(1, 0, fade_len)

        try:
            sd.play(wave.astype(np.float32), self.SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"[AudioAlert] sounddevice error: {e}")

    @staticmethod
    def _play_winsound(freq: float, duration: float):
        """Fallback Windows beep."""
        try:
            winsound.Beep(int(freq), int(duration * 1000))
        except Exception as e:
            print(f"[AudioAlert] winsound error: {e}")