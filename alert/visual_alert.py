# =============================================================================
# alert/visual_alert.py
# Visual Alert Renderer — OpenCV Overlay System
#
# Menampilkan informasi sistem secara real-time pada frame webcam:
#   - Sidebar kiri: skor, FPS, sudut kepala, status behavior
#   - Overlay tengah: warning/critical message saat ada alert
#   - Frame border: warna sesuai severity level
#   - Progress bar: Focus Score dan Suspicious Score
# =============================================================================

import cv2
import numpy as np
import time
from typing import Dict, List, Set, Tuple
from detection.head_pose import HeadPoseResult
from detection.eye_tracker import EyeTrackResult
import config.settings as cfg


class VisualAlertRenderer:
    """
    Merender semua elemen visual ke frame OpenCV.

    Layout:
        +------------------+---------------------------+
        |                  |                           |
        |    SIDEBAR       |      CAMERA FRAME         |
        |    (320px)       |      (main view)          |
        |                  |                           |
        |  - Status        |  [Head pose axes]         |
        |  - Scores        |  [Face bounding box]      |
        |  - Angles        |  [Warning overlay]        |
        |  - FPS           |                           |
        +------------------+---------------------------+
    """

    SIDEBAR_W = cfg.SIDEBAR_WIDTH

    def __init__(self, frame_w: int, frame_h: int):
        self._fw = frame_w
        self._fh = frame_h
        self._flash_until: float = 0.0  # Waktu berakhir visual flash
        print(f"[VisualAlert] Initialized — canvas {frame_w + self.SIDEBAR_W}x{frame_h}")

    # ================================================================== #
    #  Main Render Entry Point
    # ================================================================== #

    def render(self,
               camera_frame: np.ndarray,
               severity: str,
               active_behaviors: Set[str],
               durations: Dict[str, float],
               head_pose: HeadPoseResult,
               eye_track: EyeTrackResult,
               focus_score: float,
               suspicious_score: float,
               fps: float,
               warning_messages: List[str]) -> np.ndarray:
        """
        Render semua elemen visual dan kembalikan canvas lengkap.

        Args:
            camera_frame:   Frame BGR dari webcam (sudah diproses)
            severity:       "OK" | "WARNING" | "CRITICAL"
            ...             (parameter lain sesuai nama)

        Returns:
            Canvas BGR lengkap (sidebar + camera frame)
        """
        # --- Buat sidebar ---
        sidebar = self._build_sidebar(
            severity, active_behaviors, durations,
            head_pose, eye_track, focus_score, suspicious_score, fps
        )

        # --- Overlay pada camera frame ---
        frame = camera_frame.copy()
        frame = self._draw_frame_border(frame, severity)
        frame = self._draw_bounding_box_label(frame, severity)
        if warning_messages and severity != "OK":
            frame = self._draw_warning_overlay(frame, warning_messages, severity)

        # --- Gabung sidebar + frame ---
        canvas = np.hstack([sidebar, frame])
        return canvas

    # ================================================================== #
    #  Sidebar Builder
    # ================================================================== #

    def _build_sidebar(self,
                       severity: str,
                       active_behaviors: Set[str],
                       durations: Dict[str, float],
                       head_pose: HeadPoseResult,
                       eye_track: EyeTrackResult,
                       focus_score: float,
                       suspicious_score: float,
                       fps: float) -> np.ndarray:
        """Bangun panel sidebar (320 x frame_h)."""
        sb = np.full((self._fh, self.SIDEBAR_W, 3), cfg.COLOR_SIDEBAR, dtype=np.uint8)
        y = 20  # Cursor vertikal

        # --- Header ---
        y = self._put_text(sb, "FOCUS MONITOR", (10, y),
                           scale=0.65, color=cfg.COLOR_WHITE, bold=True)
        y = self._put_text(sb, "Rule-Based CV System", (10, y),
                           scale=0.42, color=cfg.COLOR_GRAY)
        y += 8
        self._hline(sb, y); y += 12

        # --- Severity Status ---
        sev_color = {
            "OK":       cfg.COLOR_OK,
            "WARNING":  cfg.COLOR_WARN,
            "CRITICAL": cfg.COLOR_CRITICAL,
        }.get(severity, cfg.COLOR_WHITE)

        # Background highlight untuk status
        cv2.rectangle(sb, (6, y - 4), (self.SIDEBAR_W - 6, y + 26),
                      tuple(c // 5 for c in sev_color), -1)
        y = self._put_text(sb, f"STATUS: {severity}", (12, y),
                           scale=0.60, color=sev_color, bold=True)
        y += 8
        self._hline(sb, y); y += 12

        # --- Focus Score Bar ---
        y = self._put_text(sb, "FOCUS SCORE", (10, y), scale=0.48, color=cfg.COLOR_GRAY)
        bar_color = (
            cfg.COLOR_OK       if focus_score >= 70 else
            cfg.COLOR_WARN     if focus_score >= 40 else
            cfg.COLOR_CRITICAL
        )
        y = self._draw_progress_bar(sb, y, focus_score, 100, bar_color, f"{focus_score:.0f}/100")
        y += 6

        # --- Suspicious Score Bar ---
        y = self._put_text(sb, "SUSPICIOUS SCORE", (10, y), scale=0.48, color=cfg.COLOR_GRAY)
        susp_color = (
            cfg.COLOR_CRITICAL if suspicious_score >= 70 else
            cfg.COLOR_WARN     if suspicious_score >= 30 else
            cfg.COLOR_OK
        )
        y = self._draw_progress_bar(sb, y, suspicious_score, 100, susp_color,
                                    f"{suspicious_score:.0f}/100")
        y += 8
        self._hline(sb, y); y += 12

        # --- Head Pose Angles ---
        y = self._put_text(sb, "HEAD POSE", (10, y), scale=0.50, color=cfg.COLOR_GRAY, bold=True)
        if head_pose.success:
            pitch_c = cfg.COLOR_WARN if abs(head_pose.pitch) > 15 else cfg.COLOR_WHITE
            yaw_c   = cfg.COLOR_WARN if abs(head_pose.yaw)   > 20 else cfg.COLOR_WHITE
            y = self._put_text(sb, f"  Pitch: {head_pose.pitch:+.1f}° (up/down)", (10, y),
                               scale=0.42, color=pitch_c)
            y = self._put_text(sb, f"  Yaw:   {head_pose.yaw:+.1f}° (L/R)", (10, y),
                               scale=0.42, color=yaw_c)
            y = self._put_text(sb, f"  Roll:  {head_pose.roll:+.1f}°", (10, y),
                               scale=0.42, color=cfg.COLOR_WHITE)
        else:
            y = self._put_text(sb, "  (no face)", (10, y), scale=0.42, color=cfg.COLOR_GRAY)
        y += 8

        # --- EAR ---
        if eye_track:
            ear_c = cfg.COLOR_WARN if eye_track.ear_avg < cfg.EAR_CLOSED_THRESHOLD else cfg.COLOR_WHITE
            y = self._put_text(sb, f"EAR: {eye_track.ear_avg:.3f}", (10, y),
                               scale=0.42, color=ear_c)
            eye_state = "CLOSED" if eye_track.eye_closed else "Open"
            y = self._put_text(sb, f"Eyes: {eye_state}", (10, y),
                               scale=0.42, color=ear_c)
        y += 6
        self._hline(sb, y); y += 12

        # --- Active Behaviors ---
        y = self._put_text(sb, "BEHAVIORS", (10, y), scale=0.50, color=cfg.COLOR_GRAY, bold=True)
        all_behaviors = list(cfg.BEHAVIOR_LABELS.keys())
        for beh in all_behaviors:
            dur  = durations.get(beh, 0.0)
            is_active = beh in active_behaviors
            label = cfg.BEHAVIOR_LABELS.get(beh, beh)
            if is_active:
                color = cfg.COLOR_CRITICAL if dur >= 5 else cfg.COLOR_WARN
                tick  = "●"
            else:
                color = cfg.COLOR_GRAY
                tick  = "○"
            dur_str = f"{dur:.1f}s" if is_active else ""
            y = self._put_text(sb, f"  {tick} {label:<18} {dur_str}", (10, y),
                               scale=0.40, color=color)
        y += 8
        self._hline(sb, y); y += 12

        # --- FPS ---
        fps_color = cfg.COLOR_OK if fps >= 20 else cfg.COLOR_WARN
        y = self._put_text(sb, f"FPS: {fps:.1f}", (10, y), scale=0.48, color=fps_color)

        # --- Footer ---
        footer = "IEEE Student Project"
        cv2.putText(sb, footer,
                    (self.SIDEBAR_W // 2 - 75, self._fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, cfg.COLOR_GRAY, 1, cv2.LINE_AA)

        return sb

    # ================================================================== #
    #  Camera Frame Overlays
    # ================================================================== #

    def _draw_frame_border(self, frame: np.ndarray, severity: str) -> np.ndarray:
        """Gambar border berwarna di tepi frame sesuai severity."""
        color = {
            "OK":       cfg.COLOR_OK,
            "WARNING":  cfg.COLOR_WARN,
            "CRITICAL": cfg.COLOR_CRITICAL,
        }.get(severity, cfg.COLOR_WHITE)
        thickness = 4 if severity != "CRITICAL" else 8
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
        return frame

    def _draw_bounding_box_label(self, frame: np.ndarray, severity: str) -> np.ndarray:
        """Tampilkan label kecil di pojok kiri atas."""
        label = "● MONITORING ACTIVE"
        color = {
            "OK":       cfg.COLOR_OK,
            "WARNING":  cfg.COLOR_WARN,
            "CRITICAL": cfg.COLOR_CRITICAL,
        }.get(severity, cfg.COLOR_WHITE)
        # Background gelap
        cv2.rectangle(frame, (5, 5), (235, 28), (20, 20, 20), -1)
        cv2.putText(frame, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        return frame

    def _draw_warning_overlay(self,
                              frame: np.ndarray,
                              messages: List[str],
                              severity: str) -> np.ndarray:
        """
        Gambar kotak peringatan di bagian bawah frame.

        Desain: background semi-transparan dengan teks warning.
        """
        h, w = frame.shape[:2]
        bg_color = (0, 0, 180) if severity == "CRITICAL" else (0, 140, 200)
        box_h    = 30 + len(messages) * 28
        y_start  = h - box_h - 10

        # Buat overlay semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, y_start), (w - 8, h - 8), bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        # Gambar teks
        y_text = y_start + 22
        for i, msg in enumerate(messages):
            scale  = 0.65 if i == 0 else 0.52
            weight = 2    if i == 0 else 1
            color  = cfg.COLOR_WHITE
            cv2.putText(frame, msg, (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
            y_text += 28

        return frame

    # ================================================================== #
    #  Helper Drawing Utilities
    # ================================================================== #

    def _put_text(self, img: np.ndarray,
                  text: str,
                  pos: Tuple[int, int],
                  scale: float = 0.45,
                  color: tuple = (255, 255, 255),
                  bold: bool = False) -> int:
        """
        Tulis teks dan kembalikan koordinat Y baris berikutnya.
        """
        thickness = 2 if bold else 1
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        line_h = int(scale * 35) + 4
        return pos[1] + line_h

    def _draw_progress_bar(self,
                           img: np.ndarray,
                           y: int,
                           value: float,
                           max_val: float,
                           color: tuple,
                           label: str = "") -> int:
        """Gambar progress bar horizontal."""
        x1, x2 = 10, self.SIDEBAR_W - 10
        bar_h   = 14
        bw      = x2 - x1
        filled  = int(bw * max(0, min(value, max_val)) / max_val)

        # Background
        cv2.rectangle(img, (x1, y), (x2, y + bar_h), (60, 60, 60), -1)
        # Fill
        if filled > 0:
            cv2.rectangle(img, (x1, y), (x1 + filled, y + bar_h), color, -1)
        # Border
        cv2.rectangle(img, (x1, y), (x2, y + bar_h), (100, 100, 100), 1)
        # Label
        cv2.putText(img, label, (x1 + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        return y + bar_h + 6

    def _hline(self, img: np.ndarray, y: int, color: tuple = (60, 60, 60)):
        """Gambar garis horizontal pemisah di sidebar."""
        cv2.line(img, (10, y), (self.SIDEBAR_W - 10, y), color, 1)