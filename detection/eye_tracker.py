# =============================================================================
# detection/eye_tracker.py
# Pelacak Mata — Eye Aspect Ratio (EAR) dan Posisi Iris
#
# Algoritma EAR:
#   EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
#
#   Di mana p1–p6 adalah 6 landmark kontur mata (searah jarum jam):
#     p1 = sudut kiri, p4 = sudut kanan,
#     p2,p3 = atas, p5,p6 = bawah
#
#   EAR mendekati 0 saat mata tertutup dan ~0.25–0.30 saat terbuka normal.
#
# Referensi IEEE:
#   Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks,"
#   CVWW, 2016.
# =============================================================================

import numpy as np
from dataclasses import dataclass
from typing import Optional
from detection.face_detector import FaceData
import config.settings as cfg


@dataclass
class EyeTrackResult:
    """Hasil analisis mata."""
    ear_left:   float           # Eye Aspect Ratio mata kiri
    ear_right:  float           # Eye Aspect Ratio mata kanan
    ear_avg:    float           # Rata-rata EAR
    eye_closed: bool            # True jika mata dianggap tertutup
    gaze_x:     float           # Posisi iris horizontal (-1.0 = kiri, +1.0 = kanan)
    gaze_y:     float           # Posisi iris vertikal   (-1.0 = atas, +1.0 = bawah)


class EyeTracker:
    """
    Menghitung Eye Aspect Ratio (EAR) dari landmark MediaPipe Face Mesh.

    MediaPipe Face Mesh menggunakan indeks landmark berbeda dari
    dlib. Indeks di bawah dipilih berdasarkan anatomi kontur mata:

    Mata Kiri  (dari perspektif kamera — mata kanan subjek):
        33  = sudut luar (temporal)
        160 = atas-kiri
        158 = atas-kanan
        133 = sudut dalam (nasal)
        153 = bawah-kanan
        144 = bawah-kiri

    Mata Kanan (dari perspektif kamera — mata kiri subjek):
        362 = sudut luar (temporal)
        385 = atas-kanan
        387 = atas-kiri
        263 = sudut dalam (nasal)
        373 = bawah-kiri
        380 = bawah-kanan
    """

    # Indeks landmark untuk setiap mata (p1..p6 sesuai formula EAR)
    LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # Indeks iris center dari MediaPipe (refine_landmarks=True)
    IRIS_LEFT_CENTER  = 468  # Pusat iris kiri
    IRIS_RIGHT_CENTER = 473  # Pusat iris kanan

    # Titik referensi sudut mata untuk kalkulasi posisi iris
    # Left eye outer/inner corner: 33, 133
    # Right eye outer/inner corner: 362, 263
    LEFT_EYE_CORNERS  = (33,  133)
    RIGHT_EYE_CORNERS = (362, 263)

    def __init__(self):
        self._consec_closed = 0     # Counter frame berturutan mata tertutup
        print("[EyeTracker] Initialized.")

    def analyze(self, face_data: FaceData) -> EyeTrackResult:
        """
        Hitung EAR dan posisi iris dari FaceData.

        Args:
            face_data: Hasil dari FaceDetector.process()

        Returns:
            EyeTrackResult
        """
        lm = face_data.landmarks_px
        lm_norm = face_data.landmarks_norm

        # -- EAR Calculation --
        ear_left  = self._compute_ear(lm, self.LEFT_EYE_IDX)
        ear_right = self._compute_ear(lm, self.RIGHT_EYE_IDX)
        ear_avg   = (ear_left + ear_right) / 2.0

        # -- Eye State --
        if ear_avg < cfg.EAR_CLOSED_THRESHOLD:
            self._consec_closed += 1
        else:
            self._consec_closed = 0

        eye_closed = self._consec_closed >= cfg.EAR_CONSEC_FRAMES

        # -- Iris Position (gaze proxy) --
        gaze_x, gaze_y = 0.0, 0.0
        if face_data.iris_left_px is not None and face_data.iris_right_px is not None:
            gaze_x, gaze_y = self._compute_gaze(lm, face_data)

        return EyeTrackResult(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            eye_closed=eye_closed,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )

    @staticmethod
    def _compute_ear(landmarks: np.ndarray, eye_indices: list) -> float:
        """
        Hitung Eye Aspect Ratio untuk satu mata.

        EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

        Args:
            landmarks: Array koordinat pixel semua landmark (468, 2)
            eye_indices: 6 indeks landmark untuk satu mata

        Returns:
            Nilai EAR (float, 0.0 jika |p1-p4| = 0)
        """
        p1, p2, p3, p4, p5, p6 = [landmarks[i].astype(float) for i in eye_indices]

        # Jarak vertikal
        vert_1 = np.linalg.norm(p2 - p6)
        vert_2 = np.linalg.norm(p3 - p5)

        # Jarak horizontal
        horiz  = np.linalg.norm(p1 - p4)

        if horiz < 1e-6:
            return 0.0

        return float((vert_1 + vert_2) / (2.0 * horiz))

    @staticmethod
    def _compute_gaze(landmarks: np.ndarray,
                      face_data: FaceData) -> tuple:
        """
        Hitung posisi relatif iris terhadap sudut mata.

        Mengembalikan (gaze_x, gaze_y) dalam rentang [-1.0, +1.0]:
          gaze_x: -1 = iris di paling kiri, +1 = iris di paling kanan
          gaze_y: -1 = iris di paling atas,  +1 = iris di paling bawah

        Ini adalah aproksimasi sederhana berbasis geometri.
        """
        # Rata-rata iris kiri dan kanan untuk mendapatkan gaze center
        iris_left  = face_data.iris_left_px[0]   # pusat iris kiri  (x, y)
        iris_right = face_data.iris_right_px[0]  # pusat iris kanan (x, y)

        # Sudut mata kiri
        lc_l = landmarks[33].astype(float)   # left eye left corner
        rc_l = landmarks[133].astype(float)  # left eye right corner

        # Sudut mata kanan
        lc_r = landmarks[362].astype(float)  # right eye left corner
        rc_r = landmarks[263].astype(float)  # right eye right corner

        def _normalized_pos(iris_pos, corner_left, corner_right):
            """Normalisasi posisi iris relatif terhadap sudut mata."""
            eye_width  = np.linalg.norm(corner_right - corner_left)
            if eye_width < 1e-6:
                return 0.0, 0.0
            offset     = iris_pos - corner_left
            gx = (np.dot(offset, corner_right - corner_left) / (eye_width**2)) * 2 - 1
            gy = (iris_pos[1] - min(corner_left[1], corner_right[1])) / eye_width
            return float(np.clip(gx, -1, 1)), float(np.clip(gy, -1, 1))

        gx_l, gy_l = _normalized_pos(iris_left,  lc_l, rc_l)
        gx_r, gy_r = _normalized_pos(iris_right, lc_r, rc_r)

        # Rata-rata gaze dari kedua mata
        return (gx_l + gx_r) / 2.0, (gy_l + gy_r) / 2.0