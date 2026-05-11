#!/usr/bin/env python3
# =============================================================================
# main.py
# Student Focus & Suspicious Behavior Monitor
# Real-Time Rule-Based Computer Vision System
#
# Judul Paper  : Sistem Analisis Fokus dan Perilaku Mencurigakan Mahasiswa
#                Secara Real-Time Menggunakan Pendekatan Rule-Based
#                Computer Vision
# Teknologi    : OpenCV, MediaPipe Face Mesh, Rule-Based Logic
# Platform     : Desktop Python (cross-platform)
# Author       : [Nama Mahasiswa]
# Institusi    : [Nama Universitas]
#
# =============================================================================
# SYSTEM FLOW
# -----------
#   Webcam → Frame
#     │
#     ├─→ FaceDetector (MediaPipe Face Mesh)
#     │       → 468 landmarks, iris landmarks
#     │
#     ├─→ HeadPoseEstimator (solvePnP)
#     │       → pitch, yaw, roll (derajat)
#     │
#     ├─→ EyeTracker (EAR + iris)
#     │       → ear_avg, eye_closed, gaze_x/y
#     │
#     ├─→ BehaviorAnalyzer (Rule-Based)
#     │       → set of active behavior IDs
#     │
#     ├─→ TemporalTracker
#     │       → durasi tiap behavior (detik)
#     │
#     ├─→ BehaviorAnalyzer.compute_severity()
#     │       → "OK" | "WARNING" | "CRITICAL"
#     │
#     ├─→ ScoreCalculator
#     │       → focus_score, suspicious_score
#     │
#     ├─→ VisualAlertRenderer
#     │       → annotated frame + sidebar
#     │
#     └─→ AudioAlert (jika severity ≠ OK)
#             → background thread beep
# =============================================================================

import sys
import time
import argparse
import cv2

# Project modules
import config.settings as cfg
from detection.face_detector  import FaceDetector
from detection.head_pose       import HeadPoseEstimator
from detection.eye_tracker     import EyeTracker
from detection.behavior_analyzer import BehaviorAnalyzer
from alert.visual_alert        import VisualAlertRenderer
from alert.audio_alert         import AudioAlert
from utils.fps_counter         import FPSCounter
from utils.temporal_tracker    import TemporalTracker
from utils.score_calculator    import ScoreCalculator
from utils.session_logger       import SessionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Student Focus & Suspicious Behavior Monitor"
    )
    parser.add_argument("--camera",   type=int, default=cfg.CAMERA_INDEX,
                        help=f"Indeks webcam (default: {cfg.CAMERA_INDEX})")
    parser.add_argument("--width",    type=int, default=cfg.FRAME_WIDTH,
                        help=f"Lebar frame (default: {cfg.FRAME_WIDTH})")
    parser.add_argument("--height",   type=int, default=cfg.FRAME_HEIGHT,
                        help=f"Tinggi frame (default: {cfg.FRAME_HEIGHT})")
    parser.add_argument("--no-audio", action="store_true",
                        help="Nonaktifkan audio alert")
    parser.add_argument("--show-lm",  action="store_true",
                        help="Tampilkan semua Face Mesh landmarks")
    return parser.parse_args()


def open_camera(camera_idx: int, width: int, height: int) -> cv2.VideoCapture:
    """Buka webcam dengan konfigurasi resolusi dan FPS."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka kamera dengan indeks {camera_idx}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.TARGET_FPS)
    # Kurangi buffer untuk mengurangi latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Opened — index={camera_idx}, resolution={actual_w}x{actual_h}")
    return cap, actual_w, actual_h


def main():
    args = parse_args()

    print("=" * 60)
    print("  Student Focus & Suspicious Behavior Monitor")
    print("  Rule-Based Computer Vision System")
    print("=" * 60)
    print(f"  Camera  : {args.camera}")
    print(f"  Audio   : {'OFF' if args.no_audio else 'ON'}")
    print(f"  Press Q or ESC to quit")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Inisialisasi Kamera
    # ------------------------------------------------------------------ #
    try:
        cap, frame_w, frame_h = open_camera(args.camera, args.width, args.height)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Inisialisasi Semua Modul
    # ------------------------------------------------------------------ #
    face_detector  = FaceDetector()
    head_pose_est  = HeadPoseEstimator(frame_w, frame_h)
    eye_tracker    = EyeTracker()
    behavior_anal  = BehaviorAnalyzer()
    visual_render  = VisualAlertRenderer(frame_w, frame_h)
    audio_alert    = AudioAlert()
    fps_counter    = FPSCounter(window_size=30)
    temporal_track = TemporalTracker()
    score_calc     = ScoreCalculator()
    session_log    = SessionLogger()

    print("\n[System] All modules initialized. Starting main loop...\n")

    # ------------------------------------------------------------------ #
    #  Variabel State
    # ------------------------------------------------------------------ #
    prev_severity   = "OK"     # Track perubahan severity untuk audio clear
    head_pose_res   = None
    eye_track_res   = None

    # ------------------------------------------------------------------ #
    #  Main Processing Loop
    # ------------------------------------------------------------------ #
    while True:
        # --- Baca Frame ---
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak dapat dibaca dari kamera.")
            time.sleep(0.05)
            continue

        # Flip horizontal (mirror effect — lebih natural untuk pengguna)
        frame = cv2.flip(frame, 1)
        fps_counter.tick()

        # ============================================================== #
        #  STEP 1: Face Detection
        # ============================================================== #
        face_present, faces = face_detector.process(frame)
        face_data = faces[0] if face_present and len(faces) > 0 else None

        # ============================================================== #
        #  STEP 2: Head Pose Estimation (jika wajah terdeteksi)
        # ============================================================== #
        if face_data is not None:
            head_pose_res = head_pose_est.estimate(face_data)
            if cfg.SHOW_POSE_AXES and head_pose_res.success:
                head_pose_est.draw_pose_axes(frame, head_pose_res, face_data)
            if args.show_lm:
                face_detector.draw_landmarks(frame, face_data, draw_tesselation=False)
        else:
            # Buat dummy result saat wajah tidak ada
            from detection.head_pose import HeadPoseResult
            import numpy as np
            head_pose_res = HeadPoseResult(0, 0, 0, np.zeros(3), np.zeros(3), False)

        # ============================================================== #
        #  STEP 3: Eye Tracking (EAR + Iris)
        # ============================================================== #
        if face_data is not None:
            eye_track_res = eye_tracker.analyze(face_data)
        else:
            from detection.eye_tracker import EyeTrackResult
            eye_track_res = EyeTrackResult(0.30, 0.30, 0.30, False, 0.0, 0.0)

        # ============================================================== #
        #  STEP 4: Rule-Based Behavior Analysis
        # ============================================================== #
        active_behaviors = behavior_anal.analyze(
            face_present=face_present,
            head_pose=head_pose_res,
            eye_track=eye_track_res,
        )

        # ============================================================== #
        #  STEP 5: Temporal Analysis (durasi tiap behavior)
        # ============================================================== #
        durations = temporal_track.update(active_behaviors)

        # ============================================================== #
        #  STEP 6: Severity Classification
        # ============================================================== #
        severity = behavior_anal.compute_severity(active_behaviors, durations)

        # Update session analytics
        session_log.update(severity)

        # ============================================================== #
        #  STEP 7: Score Update
        # ============================================================== #
        focus_score, suspicious_score = score_calc.update(
            active_behaviors, durations, severity
        )

        # ============================================================== #
        #  STEP 8: Warning Messages
        # ============================================================== #
        warning_messages = behavior_anal.get_warning_messages(
            active_behaviors, durations, severity
        )

        # ============================================================== #
        #  STEP 9: Visual Rendering
        # ============================================================== #
        canvas = visual_render.render(
            camera_frame     = frame,
            severity         = severity,
            active_behaviors = active_behaviors,
            durations        = durations,
            head_pose        = head_pose_res,
            eye_track        = eye_track_res,
            focus_score      = focus_score,
            suspicious_score = suspicious_score,
            fps              = fps_counter.fps,
            warning_messages = warning_messages,
            session_stats    = session_log.get_stats(),
        )

        # ============================================================== #
        #  STEP 10: Audio Alert
        # ============================================================== #
        if not args.no_audio:
            if severity == "CRITICAL":
                audio_alert.alert_critical(cooldown=cfg.AUDIO_COOLDOWN_CRIT)
            elif severity == "WARNING":
                audio_alert.alert_warning(cooldown=cfg.AUDIO_COOLDOWN_WARN)
            elif prev_severity != "OK" and severity == "OK":
                # Kembali normal → nada singkat
                audio_alert.alert_clear()

        prev_severity = severity

        # ============================================================== #
        #  STEP 11: Display
        # ============================================================== #
        cv2.imshow("Student Focus Monitor", canvas)

        # --- Keyboard Handler ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q atau ESC
            print("[System] Quit requested by user.")
            break
        elif key == ord('r'):                   # R → Reset scores
            score_calc.reset()
            temporal_track.reset()
            session_log.reset()
            print("[System] Scores and timers reset.")
        elif key == ord('s'):                   # S → Screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.png"
            cv2.imwrite(fname, canvas)
            print(f"[System] Screenshot saved: {fname}")

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #
    cap.release()
    face_detector.release()
    cv2.destroyAllWindows()
    print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()