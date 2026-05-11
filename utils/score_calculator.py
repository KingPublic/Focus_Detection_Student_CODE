# =============================================================================
# utils/score_calculator.py
# Focus Score & Suspicious Score Calculator
#
# PERBAIKAN v2:
#   - SUSPICIOUS_DECAY_RATE diturunkan drastis (dari 1.5 → 0.2/frame)
#     Sebelumnya score naik tapi langsung terhapus saat 1 detik normal.
#   - FOCUS_DECAY lebih proporsional terhadap durasi behavior
#   - Tambah property suspicious_level untuk label UI
# =============================================================================

from typing import Dict, Set
import config.settings as cfg


class ScoreCalculator:
    """
    Kalkulasi Focus Score (0–100) dan Suspicious Score (0–100) per frame.

    Focus Score    : Mulai 100, turun saat behavior aktif, naik saat normal
    Suspicious Score: Mulai 0, naik saat behavior aktif, turun lambat saat normal

    v2 fix: decay rate suspicious score diperlambat agar akumulasi terlihat nyata.
    """

    # --- Rate constants (override settings dengan nilai yang sudah difix) ---
    _FOCUS_DECAY    = 1.5    # per frame per behavior aktif
    _FOCUS_RECOVERY = 0.5    # per frame saat normal
    _SUSP_RISE      = 4.0    # per frame per behavior aktif
    _SUSP_DECAY     = 0.2    # per frame saat normal (lambat — supaya akumulasi terasa)

    def __init__(self):
        self.focus_score:      float = 100.0
        self.suspicious_score: float = 0.0
        print("[ScoreCalculator] Initialized (Focus=100, Suspicious=0).")

    def update(self,
               active_behaviors: Set[str],
               durations: Dict[str, float],
               severity: str) -> tuple:
        """
        Update skor berdasarkan state saat ini. Dipanggil setiap frame.

        Returns:
            (focus_score: float, suspicious_score: float)
        """
        n = len(active_behaviors)

        if n == 0:
            # Kondisi normal — recovery
            self.focus_score      = min(100.0, self.focus_score + self._FOCUS_RECOVERY)
            self.suspicious_score = max(0.0,   self.suspicious_score - self._SUSP_DECAY)

        else:
            # Multiplier berdasarkan severity
            mult = 2.0 if severity == "CRITICAL" else \
                   1.0 if severity == "WARNING"  else 0.3

            self.focus_score = max(
                0.0,
                self.focus_score - (self._FOCUS_DECAY * n * mult)
            )
            self.suspicious_score = min(
                100.0,
                self.suspicious_score + (self._SUSP_RISE * n * mult)
            )

        return self.focus_score, self.suspicious_score

    def reset(self):
        self.focus_score      = 100.0
        self.suspicious_score = 0.0

    @property
    def focus_level(self) -> str:
        if self.focus_score >= 70: return "HIGH"
        if self.focus_score >= 40: return "MEDIUM"
        return "LOW"

    @property
    def suspicious_level(self) -> str:
        if self.suspicious_score >= 70: return "HIGH"
        if self.suspicious_score >= 35: return "MEDIUM"
        return "LOW"