# =============================================================================
# detection/head_pose.py
# Head Pose Estimation menggunakan OpenCV solvePnP
#
# Algoritma:
#   1. Pilih 6 titik wajah 3D referensi (model generic/canonical face)
#   2. Cocokkan dengan 6 landmark 2D dari MediaPipe
#   3. Gunakan cv2.solvePnP (SOLVEPNP_ITERATIVE) untuk mendapatkan
#      rotation vector (rvec) dan translation vector (tvec)
#   4. Konversi rvec → rotation matrix via cv2.Rodrigues
#   5. Ekstrak sudut Euler: Pitch (atas-bawah), Yaw (kiri-kanan), Roll (miring)
#
# Referensi IEEE:
#   Zhu et al., "Face Alignment Across Large Poses: A 3D Solution,"
#   CVPR, 2016.
#   OpenCV Documentation: solvePnP, Rodrigues.
# =============================================================================

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from detection.face_detector import FaceData


@dataclass
class HeadPoseResult:
    """Hasil estimasi pose kepala."""
    pitch: float          # Sudut atas-bawah (derajat) — positif = menunduk
    yaw:   float          # Sudut kiri-kanan (derajat) — positif = ke kanan
    roll:  float          # Sudut miring     (derajat)
    rvec:  np.ndarray     # Rotation vector (Rodrigues)
    tvec:  np.ndarray     # Translation vector
    success: bool         # solvePnP berhasil atau tidak


class HeadPoseEstimator:
    """
    Estimasi pose kepala menggunakan metode PnP (Perspective-n-Point).

    Menggunakan 6 landmark wajah untuk mencocokkan model 3D generik
    dengan titik 2D yang terdeteksi MediaPipe. Hasilnya adalah sudut
    Euler (pitch, yaw, roll) yang merepresentasikan orientasi kepala.

    Landmark yang digunakan:
        - Ujung hidung (nose tip)     → Landmark #4
        - Dagu (chin)                 → Landmark #152
        - Sudut mata kiri             → Landmark #33
        - Sudut mata kanan            → Landmark #263
        - Sudut mulut kiri            → Landmark #61
        - Sudut mulut kanan           → Landmark #291
    """

    # ---- 3D Model Reference Points (Generic Face, satuan mm) ----
    # Titik-titik ini merepresentasikan wajah generik dalam ruang 3D.
    # Asal (0,0,0) berada di ujung hidung.
    FACE_3D_MODEL = np.array([
        [  0.0,    0.0,    0.0],   # Nose tip           (#4)
        [  0.0,  -63.6,  -12.5],   # Chin               (#152)
        [-43.3,   32.7,  -26.0],   # Left eye corner    (#33)
        [ 43.3,   32.7,  -26.0],   # Right eye corner   (#263)
        [-28.9,  -28.9,  -24.1],   # Left mouth corner  (#61)
        [ 28.9,  -28.9,  -24.1],   # Right mouth corner (#291)
    ], dtype=np.float64)

    # ---- Indeks MediaPipe Landmark yang berkorespondensi ----
    LANDMARK_INDICES = [4, 152, 33, 263, 61, 291]

    def __init__(self, frame_width: int, frame_height: int):
        self._w = frame_width
        self._h = frame_height
        self._camera_matrix = self._build_camera_matrix()
        self._dist_coeffs   = np.zeros((4, 1), dtype=np.float64)
        print(f"[HeadPose] Camera matrix built — {frame_width}x{frame_height}")

    def _build_camera_matrix(self) -> np.ndarray:
        """
        Bangun camera intrinsic matrix (aproksimasi).

        Focal length diperkirakan = lebar frame (heuristik umum).
        Principal point diperkirakan di tengah frame.

            K = [f  0  cx]
                [0  f  cy]
                [0  0   1]
        """
        f  = float(self._w)          # focal length approx
        cx = float(self._w) / 2.0
        cy = float(self._h) / 2.0
        return np.array([
            [f,  0,  cx],
            [0,  f,  cy],
            [0,  0,  1.0],
        ], dtype=np.float64)

    def estimate(self, face_data: FaceData) -> HeadPoseResult:
        """
        Hitung pose kepala dari landmark wajah.

        Args:
            face_data: Hasil deteksi dari FaceDetector

        Returns:
            HeadPoseResult dengan sudut pitch, yaw, roll
        """
        lm_px = face_data.landmarks_px

        # Ekstrak 6 titik 2D yang berkorespondensi dengan model 3D
        pts_2d = np.array(
            [lm_px[i] for i in self.LANDMARK_INDICES],
            dtype=np.float64
        )

        # solvePnP: temukan rvec dan tvec yang memetakan 3D → 2D
        success, rvec, tvec = cv2.solvePnP(
            self.FACE_3D_MODEL,   # titik model 3D
            pts_2d,               # titik terdeteksi 2D
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult(0, 0, 0, np.zeros(3), np.zeros(3), False)

        # Konversi rotation vector ke rotation matrix (Rodrigues formula)
        rot_mat, _ = cv2.Rodrigues(rvec)

        # Ekstrak sudut Euler dari rotation matrix (ZYX convention)
        pitch, yaw, roll = self._rot_matrix_to_euler(rot_mat)

        # Koreksi arah sumbu (konvensi intuitif: tunduk=+, kanan=+)
        pitch = -pitch
        yaw   = -yaw

        # Normalisasi "flipped solution" solvePnP
        # Saat wajah menghadap kamera, solvePnP kadang memilih solusi
        # "kepala terbalik" sehingga pitch bisa bernilai ~-175 atau +175.
        # Konversi ke representasi ekuivalen dalam range [-90, +90].
        # Contoh: pitch=-175 -> -(180-175)*(-1) = -5 deg (benar: hampir lurus)
        if pitch > 90.0:
            pitch =  180.0 - pitch
            yaw   = -yaw
            roll  =  roll - 180.0 if roll > 0 else roll + 180.0
        elif pitch < -90.0:
            pitch = -180.0 - pitch
            yaw   = -yaw
            roll  =  roll - 180.0 if roll > 0 else roll + 180.0

        return HeadPoseResult(pitch=pitch, yaw=yaw, roll=roll,
                              rvec=rvec, tvec=tvec, success=True)

    @staticmethod
    def _rot_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Ekstrak sudut Euler (pitch, yaw, roll) dari rotation matrix 3x3.

        Menggunakan dekomposisi Tait-Bryan ZYX.
        Menangani singularity (gimbal lock) jika |R[2,0]| ≈ 1.

        Returns:
            (pitch_deg, yaw_deg, roll_deg)
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6  # Cek singularitas

        if not singular:
            pitch_rad = np.arctan2( R[2, 1],  R[2, 2])
            yaw_rad   = np.arctan2(-R[2, 0],  sy)
            roll_rad  = np.arctan2( R[1, 0],  R[0, 0])
        else:
            # Gimbal lock — set roll=0
            pitch_rad = np.arctan2(-R[1, 2], R[1, 1])
            yaw_rad   = np.arctan2(-R[2, 0], sy)
            roll_rad  = 0.0

        return (
            float(np.degrees(pitch_rad)),
            float(np.degrees(yaw_rad)),
            float(np.degrees(roll_rad)),
        )

    def draw_pose_axes(self, frame: np.ndarray,
                       result: HeadPoseResult,
                       face_data: FaceData,
                       axis_length: float = 60.0) -> np.ndarray:
        """
        Gambar tiga sumbu koordinat 3D pada wajah (debugging / paper viz).

        Merah  = sumbu X (kanan)
        Hijau  = sumbu Y (bawah)
        Biru   = sumbu Z (keluar dari layar)
        """
        if not result.success:
            return frame

        # Titik asal (ujung hidung dalam 2D)
        nose_tip = face_data.landmarks_px[4]

        # Proyeksikan sumbu 3D ke 2D
        axis_3d = np.float32([
            [axis_length, 0, 0],   # X
            [0, axis_length, 0],   # Y
            [0, 0, axis_length],   # Z (positif keluar dari kamera)
        ])
        axis_2d, _ = cv2.projectPoints(
            axis_3d, result.rvec, result.tvec,
            self._camera_matrix, np.zeros((4, 1))
        )
        axis_2d = axis_2d.reshape(-1, 2).astype(int)

        origin = tuple(nose_tip)
        cv2.arrowedLine(frame, origin, tuple(axis_2d[0]), (0, 0, 220),  2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, tuple(axis_2d[1]), (0, 200, 0),  2, tipLength=0.2)
        cv2.arrowedLine(frame, origin, tuple(axis_2d[2]), (220, 0, 0),  2, tipLength=0.2)
        return frame