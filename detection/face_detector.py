# =============================================================================
# detection/face_detector.py
# Wrapper MediaPipe Face Mesh untuk deteksi wajah dan ekstraksi landmark
#
# Kompatibel dengan MediaPipe 0.10.x (versi pinned di requirements.txt)
# =============================================================================

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple
import config.settings as cfg

# --- Import MediaPipe Face Mesh ---
# Gunakan mediapipe==0.10.11 (lihat requirements.txt).
# Import langsung dari submodul untuk menghindari masalah versi.
import mediapipe
from mediapipe.python.solutions.face_mesh    import FaceMesh
from mediapipe.python.solutions.drawing_utils  import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_face_mesh_tesselation_style
from mediapipe.python.solutions import face_mesh as _mp_face_mesh_module


@dataclass
class FaceData:
    """Struktur data hasil deteksi satu wajah."""
    landmarks_px:   np.ndarray              # shape (468+, 2) — koordinat pixel (x, y)
    landmarks_norm: np.ndarray              # shape (468+, 3) — koordinat normalisasi (x,y,z)
    face_rect:      Tuple[int,int,int,int]  # (x, y, w, h) bounding box wajah
    iris_left_px:   Optional[np.ndarray]   # koordinat iris kiri  (px)
    iris_right_px:  Optional[np.ndarray]   # koordinat iris kanan (px)


class FaceDetector:
    """
    Wrapper MediaPipe Face Mesh.

    Menyediakan:
        - Deteksi ada/tidaknya wajah
        - Koordinat 468 landmark wajah dalam piksel
        - Koordinat iris (jika refine_landmarks=True)
        - Bounding box wajah
    """

    # Indeks iris (tersedia jika refine_landmarks=True)
    # Left iris: 468–472  |  Right iris: 473–477
    IRIS_LEFT_IDX  = [468, 469, 470, 471, 472]
    IRIS_RIGHT_IDX = [473, 474, 475, 476, 477]

    def __init__(self):
        self._face_mesh = FaceMesh(
            max_num_faces=cfg.MAX_NUM_FACES,
            refine_landmarks=cfg.REFINE_LANDMARKS,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        )
        print("[FaceDetector] Initialized — MediaPipe Face Mesh aktif.")

    def process(self, bgr_frame: np.ndarray) -> Tuple[bool, List[FaceData]]:
        """
        Proses satu frame BGR → deteksi wajah dan ekstraksi landmark.

        Returns:
            (face_detected: bool, faces: List[FaceData])
        """
        h, w = bgr_frame.shape[:2]

        # MediaPipe butuh RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return False, []

        faces: List[FaceData] = []
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # Koordinat normalisasi (x, y, z)
            landmarks_norm = np.array(
                [[p.x, p.y, p.z] for p in lm], dtype=np.float32
            )

            # Konversi ke koordinat pixel
            landmarks_px = (landmarks_norm[:, :2] * np.array([w, h])).astype(np.int32)

            # Bounding box dari landmark
            xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]
            x1, y1 = max(0, xs.min() - 10), max(0, ys.min() - 10)
            x2, y2 = min(w, xs.max() + 10), min(h, ys.max() + 10)
            face_rect = (x1, y1, x2 - x1, y2 - y1)

            # Iris (hanya jika refine_landmarks=True → total 478 landmark)
            iris_left = iris_right = None
            if len(lm) > 476:
                iris_left = np.array(
                    [[lm[i].x * w, lm[i].y * h] for i in self.IRIS_LEFT_IDX],
                    dtype=np.float32
                )
                iris_right = np.array(
                    [[lm[i].x * w, lm[i].y * h] for i in self.IRIS_RIGHT_IDX],
                    dtype=np.float32
                )

            faces.append(FaceData(
                landmarks_px=landmarks_px,
                landmarks_norm=landmarks_norm,
                face_rect=face_rect,
                iris_left_px=iris_left,
                iris_right_px=iris_right,
            ))

        return True, faces

    def draw_landmarks(self, bgr_frame: np.ndarray,
                       face_data: FaceData,
                       draw_tesselation: bool = False) -> np.ndarray:
        """Gambar landmark wajah pada frame (opsional / debug)."""
        if draw_tesselation:
            results = self._face_mesh.process(
                cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            )
            for mp_lm in (results.multi_face_landmarks or []):
                draw_landmarks(
                    bgr_frame,
                    mp_lm,
                    _mp_face_mesh_module.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=get_default_face_mesh_tesselation_style(),
                )
        else:
            # Gambar setiap 5 landmark saja (lebih ringan)
            for pt in face_data.landmarks_px[::5]:
                cv2.circle(bgr_frame, tuple(pt), 1, (80, 200, 80), -1)
        return bgr_frame

    def release(self):
        """Bebaskan resource MediaPipe."""
        self._face_mesh.close()
        print("[FaceDetector] Released.")