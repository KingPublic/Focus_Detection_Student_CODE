# =============================================================================
# utils/temporal_tracker.py
# Temporal Analysis — Pelacak Durasi Perilaku dengan Grace Period
#
# PERBAIKAN v2:
#   Ditambahkan grace period (0.6 detik) sebelum timer direset.
#   Ini mencegah timer kembali ke nol hanya karena deteksi flicker
#   1–2 frame (umum terjadi pada MediaPipe saat pergerakan cepat).
#
# Konsep:
#   Tanpa grace period: behavior berhenti 1 frame → timer RESET → CRITICAL
#   tidak pernah tercapai karena harus mulai dari 0 lagi.
#   Dengan grace period: behavior berhenti < 0.6 detik → timer LANJUT.
# =============================================================================

import time
from dataclasses import dataclass, field
from typing import Dict, Set

# Detik toleransi sebelum timer benar-benar direset
GRACE_PERIOD_SEC = 0.6


@dataclass
class BehaviorTimer:
    """Timer untuk satu behavior."""
    start_time:       float = 0.0   # Unix timestamp saat behavior mulai
    is_active:        bool  = False  # Apakah behavior sedang aktif
    last_seen_active: float = 0.0   # Kapan terakhir behavior aktif


class TemporalTracker:
    """
    Melacak durasi setiap perilaku mencurigakan secara real-time.

    v2 — ditambahkan grace period agar timer tidak langsung reset
    saat deteksi flicker (misal: MediaPipe kehilangan tracking 1 frame).

    Prinsip kerja:
        1. Behavior aktif pertama kali → catat start_time
        2. Tiap frame behavior aktif → update last_seen_active
        3. Behavior tidak aktif TAPI masih dalam grace period → timer LANJUT
        4. Behavior tidak aktif > grace period → timer reset
    """

    def __init__(self, grace_period: float = GRACE_PERIOD_SEC):
        self._timers: Dict[str, BehaviorTimer] = {}
        self._grace  = grace_period
        print(f"[TemporalTracker] Initialized — grace period: {grace_period}s")

    def update(self, active_behaviors: Set[str]) -> Dict[str, float]:
        """
        Perbarui timer semua behavior dan kembalikan durasi aktif.

        Args:
            active_behaviors: Set ID behavior yang aktif frame ini

        Returns:
            Dict {behavior_id: duration_seconds} — hanya yang sedang aktif
            (termasuk yang masih dalam grace period)
        """
        now = time.monotonic()
        durations: Dict[str, float] = {}

        # --- Aktifkan / perpanjang timer behavior yang aktif ---
        for beh_id in active_behaviors:
            if beh_id not in self._timers:
                self._timers[beh_id] = BehaviorTimer()

            timer = self._timers[beh_id]
            if not timer.is_active:
                # Behavior baru (atau kembali setelah > grace period)
                timer.start_time = now
                timer.is_active  = True

            timer.last_seen_active = now
            durations[beh_id] = now - timer.start_time

        # --- Evaluasi behavior yang tidak aktif frame ini ---
        for beh_id, timer in self._timers.items():
            if beh_id not in active_behaviors and timer.is_active:
                gap = now - timer.last_seen_active
                if gap <= self._grace:
                    # Masih dalam grace period → anggap masih aktif
                    durations[beh_id] = now - timer.start_time
                else:
                    # Grace period habis → reset timer
                    timer.is_active  = False
                    timer.start_time = 0.0

        return durations

    def get_duration(self, behavior_id: str) -> float:
        """Kembalikan durasi aktif untuk satu behavior (0 jika tidak aktif)."""
        timer = self._timers.get(behavior_id)
        if timer is None or not timer.is_active:
            return 0.0
        return time.monotonic() - timer.start_time

    def reset(self, behavior_id: str = None):
        """Reset timer. Jika behavior_id=None → reset semua."""
        if behavior_id:
            if behavior_id in self._timers:
                t = self._timers[behavior_id]
                t.is_active = False
                t.start_time = 0.0
        else:
            self._timers.clear()