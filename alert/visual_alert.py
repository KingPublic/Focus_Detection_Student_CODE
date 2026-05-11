# =============================================================================
# alert/visual_alert.py
# Visual Alert Renderer — OpenCV Overlay System  v3
#
# Perubahan v3:
#   - Panel Overall Session Score di bagian atas sidebar
#   - Tampilkan Grade (A/B/C/D/F), durasi sesi, breakdown waktu
#   - Layout sidebar diatur ulang: Overall Score → Status → Scores → ...
# =============================================================================

import cv2
import numpy as np
import time
from typing import Dict, List, Set, Tuple, Optional
from detection.head_pose import HeadPoseResult
from detection.eye_tracker import EyeTrackResult
from utils.session_logger import SessionStats
import config.settings as cfg


class VisualAlertRenderer:
    """
    Merender semua elemen visual ke frame OpenCV.

    Layout Sidebar (atas ke bawah):
        ┌──────────────────────────┐
        │  FOCUS MONITOR / header  │
        ├──────────────────────────┤
        │  OVERALL SESSION SCORE   │  ← Baru v3
        │  Grade + durasi + bars   │
        ├──────────────────────────┤
        │  STATUS: OK/WARN/CRIT    │
        ├──────────────────────────┤
        │  Focus Score bar         │
        │  Suspicious Score bar    │
        ├──────────────────────────┤
        │  Head Pose angles        │
        │  EAR / Eye state         │
        ├──────────────────────────┤
        │  Behaviors list          │
        ├──────────────────────────┤
        │  FPS                     │
        └──────────────────────────┘
    """

    SIDEBAR_W = cfg.SIDEBAR_WIDTH

    def __init__(self, frame_w: int, frame_h: int):
        self._fw = frame_w
        self._fh = frame_h
        print(f"[VisualAlert] Initialized — canvas {frame_w + self.SIDEBAR_W}x{frame_h}")

    # ================================================================== #
    #  Main Render Entry Point
    # ================================================================== #

    def render(self,
               camera_frame:     np.ndarray,
               severity:         str,
               active_behaviors: Set[str],
               durations:        Dict[str, float],
               head_pose:        HeadPoseResult,
               eye_track:        EyeTrackResult,
               focus_score:      float,
               suspicious_score: float,
               fps:              float,
               warning_messages: List[str],
               session_stats:    Optional[SessionStats] = None) -> np.ndarray:

        sidebar = self._build_sidebar(
            severity, active_behaviors, durations,
            head_pose, eye_track, focus_score, suspicious_score,
            fps, session_stats
        )

        frame = camera_frame.copy()
        frame = self._draw_frame_border(frame, severity)
        frame = self._draw_status_badge(frame, severity, session_stats)
        if warning_messages and severity != "OK":
            frame = self._draw_warning_overlay(frame, warning_messages, severity)

        return np.hstack([sidebar, frame])

    # ================================================================== #
    #  Sidebar Builder
    # ================================================================== #

    def _build_sidebar(self, severity, active_behaviors, durations,
                       head_pose, eye_track, focus_score, suspicious_score,
                       fps, session_stats):

        sb = np.full((self._fh, self.SIDEBAR_W, 3),
                     cfg.COLOR_SIDEBAR, dtype=np.uint8)
        y = 18

        # ── Header ────────────────────────────────────────────────────
        y = self._text(sb, "FOCUS MONITOR", (10, y),
                       scale=0.65, color=cfg.COLOR_WHITE, bold=True)
        y = self._text(sb, "Rule-Based CV System", (10, y),
                       scale=0.40, color=cfg.COLOR_GRAY)
        y += 4
        self._hline(sb, y); y += 10

        # ── OVERALL SESSION SCORE (panel utama untuk dosen) ───────────
        if session_stats is not None:
            y = self._draw_overall_panel(sb, y, session_stats)
            self._hline(sb, y); y += 10

        # ── Status severity ───────────────────────────────────────────
        sev_color = {"OK": cfg.COLOR_OK,
                     "WARNING": cfg.COLOR_WARN,
                     "CRITICAL": cfg.COLOR_CRITICAL}.get(severity, cfg.COLOR_WHITE)
        cv2.rectangle(sb, (6, y - 3), (self.SIDEBAR_W - 6, y + 24),
                      tuple(c // 5 for c in sev_color), -1)
        y = self._text(sb, f"STATUS: {severity}", (12, y),
                       scale=0.58, color=sev_color, bold=True)
        y += 6
        self._hline(sb, y); y += 10

        # ── Real-time scores ──────────────────────────────────────────
        y = self._text(sb, "FOCUS SCORE", (10, y),
                       scale=0.44, color=cfg.COLOR_GRAY)
        bar_c = (cfg.COLOR_OK if focus_score >= 70 else
                 cfg.COLOR_WARN if focus_score >= 40 else cfg.COLOR_CRITICAL)
        y = self._progress_bar(sb, y, focus_score, 100, bar_c,
                               f"{focus_score:.0f}/100")
        y += 4

        y = self._text(sb, "SUSPICIOUS SCORE", (10, y),
                       scale=0.44, color=cfg.COLOR_GRAY)
        susp_c = (cfg.COLOR_CRITICAL if suspicious_score >= 70 else
                  cfg.COLOR_WARN     if suspicious_score >= 30 else cfg.COLOR_OK)
        y = self._progress_bar(sb, y, suspicious_score, 100, susp_c,
                               f"{suspicious_score:.0f}/100")
        y += 6
        self._hline(sb, y); y += 10

        # ── Head Pose ─────────────────────────────────────────────────
        y = self._text(sb, "HEAD POSE", (10, y),
                       scale=0.46, color=cfg.COLOR_GRAY, bold=True)
        if head_pose.success:
            pc = cfg.COLOR_WARN if abs(head_pose.pitch) > 15 else cfg.COLOR_WHITE
            yc = cfg.COLOR_WARN if abs(head_pose.yaw)   > 20 else cfg.COLOR_WHITE
            y = self._text(sb, f"  Pitch: {head_pose.pitch:+.1f}  (up/down)",
                           (10, y), scale=0.40, color=pc)
            y = self._text(sb, f"  Yaw:   {head_pose.yaw:+.1f}  (L/R)",
                           (10, y), scale=0.40, color=yc)
            y = self._text(sb, f"  Roll:  {head_pose.roll:+.1f}",
                           (10, y), scale=0.40, color=cfg.COLOR_WHITE)
        else:
            y = self._text(sb, "  (no face)", (10, y),
                           scale=0.40, color=cfg.COLOR_GRAY)
        y += 4

        if eye_track:
            ec = cfg.COLOR_WARN if eye_track.ear_avg < cfg.EAR_CLOSED_THRESHOLD \
                 else cfg.COLOR_WHITE
            y = self._text(sb, f"EAR: {eye_track.ear_avg:.3f}  "
                               f"Eyes: {'CLOSED' if eye_track.eye_closed else 'Open'}",
                           (10, y), scale=0.40, color=ec)
        y += 4
        self._hline(sb, y); y += 10

        # ── Behaviors ─────────────────────────────────────────────────
        y = self._text(sb, "BEHAVIORS", (10, y),
                       scale=0.46, color=cfg.COLOR_GRAY, bold=True)
        for beh, label in cfg.BEHAVIOR_LABELS.items():
            dur       = durations.get(beh, 0.0)
            is_active = beh in active_behaviors
            if is_active:
                color  = cfg.COLOR_CRITICAL if dur >= 5 else cfg.COLOR_WARN
                marker = "[*]"
                dur_s  = f"{dur:.1f}s"
            else:
                color  = cfg.COLOR_GRAY
                marker = "[ ]"
                dur_s  = ""
            line = f"  {marker} {label:<18} {dur_s}"
            y = self._text(sb, line, (10, y), scale=0.38, color=color)
        y += 6
        self._hline(sb, y); y += 8

        # ── FPS ───────────────────────────────────────────────────────
        fps_c = cfg.COLOR_OK if fps >= 20 else cfg.COLOR_WARN
        self._text(sb, f"FPS: {fps:.1f}", (10, y), scale=0.46, color=fps_c)

        # Footer
        cv2.putText(sb, "IEEE Student Project",
                    (self.SIDEBAR_W // 2 - 72, self._fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, cfg.COLOR_GRAY, 1,
                    cv2.LINE_AA)
        return sb

    # ================================================================== #
    #  Overall Session Score Panel
    # ================================================================== #

    def _draw_overall_panel(self, sb: np.ndarray, y: int,
                            stats: SessionStats) -> int:
        """
        Panel Overall Session Score — informasi utama untuk dosen/guru.

        Menampilkan:
          - Label "OVERALL SESSION SCORE"
          - Score numerik besar + Grade (A/B/C/D/F) dengan warna
          - Durasi sesi
          - Progress bar overall score
          - Breakdown waktu: Focus% / Warn% / Crit%
          - Jumlah warning & critical events
        """
        # Warna grade
        grade_color = {
            "A": (0, 220, 0),
            "B": (0, 200, 120),
            "C": (0, 200, 220),
            "D": (0, 140, 255),
            "F": (0, 30,  220),
        }.get(stats.grade, cfg.COLOR_WHITE)

        # Background panel tipis
        panel_top = y - 4
        # (gambar background setelah tahu tinggi panel)

        # -- Label --
        y = self._text(sb, "OVERALL SESSION SCORE", (10, y),
                       scale=0.48, color=cfg.COLOR_WHITE, bold=True)

        # -- Score besar + Grade sejajar --
        score_str = f"{stats.overall_score:.1f}/100"
        grade_str = f"Grade: {stats.grade}"

        cv2.putText(sb, score_str, (10, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    grade_color, 2, cv2.LINE_AA)
        cv2.putText(sb, grade_str, (165, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    grade_color, 2, cv2.LINE_AA)
        y += 26

        # -- Progress bar overall --
        y = self._progress_bar(sb, y, stats.overall_score, 100,
                               grade_color, "")
        y += 4

        # -- Durasi sesi --
        y = self._text(sb, f"  Duration : {stats.duration_str}",
                       (10, y), scale=0.40, color=cfg.COLOR_GRAY)

        # -- Breakdown waktu --
        y = self._text(sb,
            f"  Focus {stats.pct_ok:.0f}%  "
            f"Warn {stats.pct_warning:.0f}%  "
            f"Crit {stats.pct_critical:.0f}%",
            (10, y), scale=0.40, color=cfg.COLOR_GRAY)

        # -- Mini bar breakdown (tiga warna dalam satu bar) --
        y = self._draw_tricolor_bar(sb, y, stats)
        y += 2

        # -- Event counters --
        wc = cfg.COLOR_WARN     if stats.warning_events  > 0 else cfg.COLOR_GRAY
        cc = cfg.COLOR_CRITICAL if stats.critical_events > 0 else cfg.COLOR_GRAY
        y = self._text(sb,
            f"  Warn events: {stats.warning_events}   "
            f"Crit events: {stats.critical_events}",
            (10, y), scale=0.38, color=cfg.COLOR_WHITE)
        y += 4

        return y

    def _draw_tricolor_bar(self, sb: np.ndarray, y: int,
                           stats: SessionStats) -> int:
        """Bar tiga warna: hijau=fokus | kuning=warning | merah=critical."""
        x1, x2 = 10, self.SIDEBAR_W - 10
        bw, bh  = x2 - x1, 10
        total   = stats.pct_ok + stats.pct_warning + stats.pct_critical
        if total < 0.1:
            total = 100.0

        w_ok   = int(bw * stats.pct_ok       / total)
        w_warn = int(bw * stats.pct_warning  / total)
        w_crit = bw - w_ok - w_warn   # sisa agar total tepat

        cv2.rectangle(sb, (x1,           y), (x1 + w_ok,            y + bh), cfg.COLOR_OK,       -1)
        cv2.rectangle(sb, (x1 + w_ok,    y), (x1 + w_ok + w_warn,   y + bh), cfg.COLOR_WARN,     -1)
        cv2.rectangle(sb, (x1 + w_ok + w_warn, y), (x2,             y + bh), cfg.COLOR_CRITICAL, -1)
        cv2.rectangle(sb, (x1, y), (x2, y + bh), (100, 100, 100), 1)
        return y + bh + 4

    # ================================================================== #
    #  Camera Frame Overlays
    # ================================================================== #

    def _draw_frame_border(self, frame, severity):
        color = {"OK": cfg.COLOR_OK, "WARNING": cfg.COLOR_WARN,
                 "CRITICAL": cfg.COLOR_CRITICAL}.get(severity, cfg.COLOR_WHITE)
        t = 8 if severity == "CRITICAL" else 4
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, t)
        return frame

    def _draw_status_badge(self, frame, severity, session_stats):
        """Badge di pojok kiri atas — tampilkan overall score juga."""
        h, w = frame.shape[:2]
        color = {"OK": cfg.COLOR_OK, "WARNING": cfg.COLOR_WARN,
                 "CRITICAL": cfg.COLOR_CRITICAL}.get(severity, cfg.COLOR_WHITE)

        # Baris 1: status monitoring
        cv2.rectangle(frame, (5, 5), (300, 28), (20, 20, 20), -1)
        cv2.putText(frame, "* MONITORING ACTIVE", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

        # Baris 2: overall score (pojok kanan atas)
        if session_stats is not None:
            grade_color = {
                "A": (0, 220, 0), "B": (0, 200, 120), "C": (0, 200, 220),
                "D": (0, 140, 255), "F": (0, 30, 220),
            }.get(session_stats.grade, cfg.COLOR_WHITE)
            label = (f"Session: {session_stats.overall_score:.0f}/100  "
                     f"Grade:{session_stats.grade}  "
                     f"{session_stats.duration_str}")
            tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[:2]
            x_start = w - tw[0] - 14 if isinstance(tw, tuple) else w - 340
            # simpler: fixed right-align estimate
            cv2.rectangle(frame, (w - 370, 5), (w - 5, 28), (20, 20, 20), -1)
            cv2.putText(frame, label, (w - 366, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, grade_color, 1,
                        cv2.LINE_AA)
        return frame

    def _draw_warning_overlay(self, frame, messages, severity):
        h, w = frame.shape[:2]
        bg   = (0, 0, 180) if severity == "CRITICAL" else (0, 140, 200)
        bh   = 30 + len(messages) * 28
        y0   = h - bh - 10
        ov   = frame.copy()
        cv2.rectangle(ov, (8, y0), (w - 8, h - 8), bg, -1)
        frame = cv2.addWeighted(ov, 0.75, frame, 0.25, 0)
        yt = y0 + 22
        for i, msg in enumerate(messages):
            sc = 0.65 if i == 0 else 0.52
            wt = 2    if i == 0 else 1
            cv2.putText(frame, msg, (20, yt),
                        cv2.FONT_HERSHEY_SIMPLEX, sc,
                        cfg.COLOR_WHITE, wt, cv2.LINE_AA)
            yt += 28
        return frame

    # ================================================================== #
    #  Helpers
    # ================================================================== #

    def _text(self, img, text, pos, scale=0.45,
              color=(255, 255, 255), bold=False) -> int:
        t = 2 if bold else 1
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, t, cv2.LINE_AA)
        return pos[1] + int(scale * 36) + 4

    def _progress_bar(self, img, y, val, max_val, color, label="") -> int:
        x1, x2 = 10, self.SIDEBAR_W - 10
        bh      = 13
        bw      = x2 - x1
        filled  = int(bw * max(0, min(val, max_val)) / max(max_val, 1))
        cv2.rectangle(img, (x1, y), (x2, y + bh), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(img, (x1, y), (x1 + filled, y + bh), color, -1)
        cv2.rectangle(img, (x1, y), (x2, y + bh), (100, 100, 100), 1)
        if label:
            cv2.putText(img, label, (x1 + 4, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return y + bh + 5

    def _hline(self, img, y, color=(60, 60, 60)):
        cv2.line(img, (10, y), (self.SIDEBAR_W - 10, y), color, 1)