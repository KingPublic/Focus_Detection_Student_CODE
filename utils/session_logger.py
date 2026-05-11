# =============================================================================
# utils/session_logger.py
# Session Analytics & Overall Score Calculator  v2
#
# PERUBAHAN v2:
#   Overall Score sekarang HANYA BISA TURUN — tidak pernah naik kembali.
#   Menggunakan sistem akumulasi penalti per frame:
#     - Frame WARNING  → kurangi 0.008 poin
#     - Frame CRITICAL → kurangi 0.025 poin
#   Ini mencerminkan prinsip pengawasan ujian:
#   pelanggaran yang sudah terjadi tidak bisa "dihapus" oleh perilaku baik.
#
# Formula:
#   overall_score -= PENALTY_WARNING  (saat severity == WARNING)
#   overall_score -= PENALTY_CRITICAL (saat severity == CRITICAL)
#   overall_score  = max(0, overall_score)   ← tidak pernah negatif
#   overall_score tidak pernah naik          ← hanya bisa turun
# =============================================================================

import time
from dataclasses import dataclass


# Penalti per frame (disesuaikan dengan ~30 FPS)
# WARNING  : 0.008 × 30fps = 0.24 poin/detik turun
# CRITICAL : 0.025 × 30fps = 0.75 poin/detik turun
# Contoh: 10 detik CRITICAL = turun 7.5 poin
PENALTY_WARNING  = 0.008
PENALTY_CRITICAL = 0.025


@dataclass
class SessionStats:
    """Snapshot statistik sesi saat ini."""
    duration_sec:    float
    frames_total:    int
    frames_ok:       int
    frames_warning:  int
    frames_critical: int
    pct_ok:          float
    pct_warning:     float
    pct_critical:    float
    warning_events:  int
    critical_events: int
    overall_score:   float   # hanya turun, tidak pernah naik
    grade:           str
    duration_str:    str


class SessionLogger:
    """
    Merekam statistik sesi dan menghitung Overall Score.

    Overall Score = akumulasi penalti (hanya turun):
        Tiap frame WARNING  → -0.008 poin
        Tiap frame CRITICAL → -0.025 poin
        Tiap frame OK       → tidak berubah (tidak naik)

    Grade dihitung dari Overall Score akhir:
        A: 90–100 | B: 75–89 | C: 60–74 | D: 45–59 | F: 0–44
    """

    GRADE_THRESHOLDS = [(90,"A"),(75,"B"),(60,"C"),(45,"D"),(0,"F")]

    def __init__(self):
        self._start_time    = time.monotonic()
        self._frames_total  = 0
        self._frames_ok     = 0
        self._frames_warn   = 0
        self._frames_crit   = 0
        self._warn_events   = 0
        self._crit_events   = 0
        self._prev_severity = "OK"
        self._overall_score = 100.0   # hanya bisa turun
        print("[SessionLogger] Session started.")

    def update(self, severity: str):
        """Dipanggil sekali per frame dari main loop."""
        self._frames_total += 1

        if severity == "OK":
            self._frames_ok += 1
            # Skor TIDAK naik — lewati saja

        elif severity == "WARNING":
            self._frames_warn += 1
            self._overall_score = max(0.0,
                self._overall_score - PENALTY_WARNING)

        else:  # CRITICAL
            self._frames_crit += 1
            self._overall_score = max(0.0,
                self._overall_score - PENALTY_CRITICAL)

        # Hitung event transisi
        if severity == "WARNING"  and self._prev_severity != "WARNING":
            self._warn_events += 1
        if severity == "CRITICAL" and self._prev_severity != "CRITICAL":
            self._crit_events += 1

        self._prev_severity = severity

    def get_stats(self) -> SessionStats:
        """Kembalikan snapshot statistik sesi saat ini."""
        n = max(self._frames_total, 1)

        pct_ok   = self._frames_ok   / n * 100
        pct_warn = self._frames_warn / n * 100
        pct_crit = self._frames_crit / n * 100

        # Grade dari overall score
        grade = "F"
        for threshold, g in self.GRADE_THRESHOLDS:
            if self._overall_score >= threshold:
                grade = g
                break

        # Format durasi
        elapsed = time.monotonic() - self._start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        return SessionStats(
            duration_sec    = elapsed,
            frames_total    = self._frames_total,
            frames_ok       = self._frames_ok,
            frames_warning  = self._frames_warn,
            frames_critical = self._frames_crit,
            pct_ok          = pct_ok,
            pct_warning     = pct_warn,
            pct_critical    = pct_crit,
            warning_events  = self._warn_events,
            critical_events = self._crit_events,
            overall_score   = self._overall_score,
            grade           = grade,
            duration_str    = f"{h:02d}:{m:02d}:{s:02d}",
        )

    def reset(self):
        self.__init__()
        print("[SessionLogger] Session reset.")