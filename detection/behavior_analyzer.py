# =============================================================================
# detection/behavior_analyzer.py
# Rule-Based Behavior Analysis Engine
#
# Algoritma:
#   Menggunakan pendekatan rule-based murni (tanpa ML/training):
#   Setiap perilaku didefinisikan sebagai serangkaian IF-THEN rule
#   berdasarkan output dari head pose estimator dan eye tracker.
#
#   Rule dapat diperluas dengan menambahkan entri baru di BEHAVIOR_RULES.
#   Desain ini memudahkan interpretasi dan penjelasan di paper IEEE.
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Set
from detection.head_pose import HeadPoseResult
from detection.eye_tracker import EyeTrackResult
import config.settings as cfg


@dataclass
class BehaviorState:
    """Kondisi perilaku yang terdeteksi saat ini."""
    active_behaviors:   Set[str]     = field(default_factory=set)
    # Peta behavior → durasi aktif saat ini (detik, diisi oleh TemporalTracker)
    durations:          Dict[str, float] = field(default_factory=dict)
    # Severity level: "OK" | "WARNING" | "CRITICAL"
    severity:           str          = "OK"
    face_present:       bool         = True


# =============================================================================
# DEFINISI RULE
# Setiap rule adalah dict dengan kunci:
#   "id"        : identifier unik (str)
#   "label"     : label tampilan (str)
#   "check_fn"  : fungsi lambda(HeadPoseResult, EyeTrackResult) -> bool
# =============================================================================
BEHAVIOR_RULES = [
    {
        "id":    "LOOKING_LEFT",
        "label": "Looking Left",
        # Karena frame di-flip horizontal (cv2.flip), yaw positif = kepala
        # pengguna menoleh ke KIRI (kiri dari sudut pandang pengguna sendiri)
        "check_fn": lambda hp, et: (
            hp.success and hp.yaw > cfg.YAW_THRESHOLD_RIGHT
        ),
    },
    {
        "id":    "LOOKING_RIGHT",
        "label": "Looking Right",
        "check_fn": lambda hp, et: (
            hp.success and hp.yaw < cfg.YAW_THRESHOLD_LEFT
        ),
    },
    {
        "id":    "LOOKING_DOWN",
        "label": "Looking Down",
        "check_fn": lambda hp, et: (
            hp.success and hp.pitch > cfg.PITCH_DOWN_THRESHOLD
        ),
    },
    {
        "id":    "LOOKING_UP",
        "label": "Looking Up",
        "check_fn": lambda hp, et: (
            hp.success and hp.pitch < cfg.PITCH_UP_THRESHOLD
        ),
    },
    {
        "id":    "EYES_CLOSED",
        "label": "Eyes Closed",
        "check_fn": lambda hp, et: et.eye_closed,
    },
    {
        "id":    "EYE_LOOK_LEFT",
        "label": "Eye Gaze Left",
        # Deteksi melirik ke kiri menggunakan posisi iris relatif terhadap
        # sudut mata (gaze_x < threshold negatif).
        # Rule ini aktif HANYA jika kepala masih lurus (|yaw| < 15 deg)
        # agar tidak overlap dengan LOOKING_LEFT (head pose).
        "check_fn": lambda hp, et: (
            et.gaze_x < cfg.GAZE_LEFT_THRESHOLD
            and (not hp.success or abs(hp.yaw) < 15.0)
            and not et.eye_closed
        ),
    },
    {
        "id":    "EYE_LOOK_RIGHT",
        "label": "Eye Gaze Right",
        "check_fn": lambda hp, et: (
            et.gaze_x > cfg.GAZE_RIGHT_THRESHOLD
            and (not hp.success or abs(hp.yaw) < 15.0)
            and not et.eye_closed
        ),
    },
]


class BehaviorAnalyzer:
    """
    Rule-Based Behavior Analyzer.

    Mengevaluasi setiap rule dari BEHAVIOR_RULES secara berurutan
    terhadap output HeadPoseResult dan EyeTrackResult.

    Deteksi FACE_ABSENT dilakukan di level terpisah (tidak butuh hp/et).

    Prinsip desain:
        - Stateless: tidak menyimpan temporal state (dilakukan oleh TemporalTracker)
        - Extensible: tambahkan rule baru cukup di BEHAVIOR_RULES
        - Interpretable: setiap rule dapat dijelaskan dengan bahasa natural
    """

    def __init__(self):
        print(f"[BehaviorAnalyzer] {len(BEHAVIOR_RULES)} rules loaded.")

    def analyze(self,
                face_present: bool,
                head_pose: HeadPoseResult,
                eye_track: EyeTrackResult) -> Set[str]:
        """
        Evaluasi semua rule dan kembalikan set perilaku yang aktif.

        Args:
            face_present: True jika wajah terdeteksi
            head_pose:    Hasil head pose estimation
            eye_track:    Hasil eye tracking

        Returns:
            Set ID perilaku yang saat ini aktif (kosong = semua normal)
        """
        active: Set[str] = set()

        # Rule khusus: wajah tidak terdeteksi
        if not face_present:
            active.add("FACE_ABSENT")
            return active  # Tidak perlu cek rule lain jika wajah hilang

        # Evaluasi setiap rule
        for rule in BEHAVIOR_RULES:
            try:
                if rule["check_fn"](head_pose, eye_track):
                    active.add(rule["id"])
            except Exception:
                # Rule gagal → lewati (defensive programming)
                pass

        return active

    def compute_severity(self,
                         active_behaviors: Set[str],
                         durations: Dict[str, float]) -> str:
        """
        Tentukan severity level berdasarkan perilaku aktif dan durasinya.

        Severity hierarchy:
            CRITICAL > WARNING > OK

        Logic:
            - CRITICAL: ada behavior yg durasinya melewati threshold kritis
            - WARNING:  ada behavior yg durasinya melewati threshold warning
            - OK:       tidak ada behavior mencurigakan / durasi masih aman

        Returns:
            "OK" | "WARNING" | "CRITICAL"
        """
        # Mapping behavior ke threshold durasi
        WARN_THRESH = {
            "LOOKING_LEFT":   cfg.DURATION_WARN_LOOK_AWAY,
            "LOOKING_RIGHT":  cfg.DURATION_WARN_LOOK_AWAY,
            "LOOKING_DOWN":   cfg.DURATION_WARN_LOOK_DOWN,
            "LOOKING_UP":     cfg.DURATION_WARN_LOOK_AWAY,
            "FACE_ABSENT":    cfg.DURATION_WARN_FACE_ABSENT,
            "EYES_CLOSED":    cfg.DURATION_WARN_EYES_CLOSED,
            "EYE_LOOK_LEFT":  cfg.DURATION_WARN_GAZE,
            "EYE_LOOK_RIGHT": cfg.DURATION_WARN_GAZE,
        }
        CRIT_THRESH = {
            "LOOKING_LEFT":   cfg.DURATION_CRIT_LOOK_AWAY,
            "LOOKING_RIGHT":  cfg.DURATION_CRIT_LOOK_AWAY,
            "LOOKING_DOWN":   cfg.DURATION_CRIT_LOOK_DOWN,
            "LOOKING_UP":     cfg.DURATION_CRIT_LOOK_AWAY,
            "FACE_ABSENT":    cfg.DURATION_CRIT_FACE_ABSENT,
            "EYES_CLOSED":    cfg.DURATION_CRIT_EYES_CLOSED,
            "EYE_LOOK_LEFT":  cfg.DURATION_CRIT_GAZE,
            "EYE_LOOK_RIGHT": cfg.DURATION_CRIT_GAZE,
        }

        severity = "OK"
        for beh in active_behaviors:
            dur = durations.get(beh, 0.0)
            crit_t = CRIT_THRESH.get(beh, 999)
            warn_t = WARN_THRESH.get(beh, 999)

            if dur >= crit_t:
                return "CRITICAL"          # Langsung kembalikan CRITICAL
            elif dur >= warn_t:
                severity = "WARNING"       # Catat WARNING, lanjut cek lain

        return severity

    def get_warning_messages(self,
                             active_behaviors: Set[str],
                             durations: Dict[str, float],
                             severity: str) -> List[str]:
        """
        Buat daftar pesan warning yang akan ditampilkan di UI.

        Returns:
            List string pesan (bisa kosong jika OK)
        """
        if severity == "OK":
            return []

        msgs = []
        label_map = cfg.BEHAVIOR_LABELS

        for beh in sorted(active_behaviors):
            dur  = durations.get(beh, 0.0)
            lbl  = label_map.get(beh, beh)
            msgs.append(f"{lbl}: {dur:.1f}s")

        if severity == "CRITICAL":
            msgs.insert(0, "!! SUSPICIOUS BEHAVIOR DETECTED")
        else:
            msgs.insert(0, "!! ATTENTION REQUIRED")

        return msgs