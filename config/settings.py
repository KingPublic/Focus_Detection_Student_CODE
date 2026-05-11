# =============================================================================
# config/settings.py
# Centralized configuration for Student Focus Monitor  v2
# =============================================================================

# --- CAMERA ---
CAMERA_INDEX        = 0
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
TARGET_FPS          = 30

# --- MEDIAPIPE ---
MAX_NUM_FACES               = 1
MIN_DETECTION_CONFIDENCE    = 0.5
MIN_TRACKING_CONFIDENCE     = 0.5
REFINE_LANDMARKS            = True

# --- HEAD POSE THRESHOLDS (derajat) ---
YAW_THRESHOLD_LEFT    = -20.0
YAW_THRESHOLD_RIGHT   =  20.0
PITCH_DOWN_THRESHOLD  =  15.0
PITCH_UP_THRESHOLD    = -18.0

# --- EAR ---
EAR_CLOSED_THRESHOLD  = 0.20
EAR_CONSEC_FRAMES     = 3

# --- TEMPORAL THRESHOLDS (detik) ---
# WARNING  = mulai tampilkan alert kuning
# CRITICAL = alert merah + audio kencang
DURATION_WARN_LOOK_AWAY  = 3.0
DURATION_WARN_LOOK_DOWN  = 4.0
DURATION_WARN_FACE_ABSENT= 2.0
DURATION_WARN_EYES_CLOSED= 3.0

DURATION_CRIT_LOOK_AWAY  = 5.0   # diturunkan dari 7 → 5 detik
DURATION_CRIT_LOOK_DOWN  = 6.0   # diturunkan dari 8 → 6 detik
DURATION_CRIT_FACE_ABSENT= 4.0   # diturunkan dari 5 → 4 detik
DURATION_CRIT_EYES_CLOSED= 5.0   # diturunkan dari 6 → 5 detik

# --- SCORE (digunakan ScoreCalculator internal, tidak di-override settings) ---
FOCUS_SCORE_INIT        = 100.0
FOCUS_DECAY_PER_BEHAVIOR=   1.5
FOCUS_RECOVERY_RATE     =   0.5
SUSPICIOUS_INIT         =   0.0
SUSPICIOUS_RISE_RATE    =   4.0
SUSPICIOUS_DECAY_RATE   =   0.2   # FIX: diturunkan dari 1.5 → 0.2

# --- AUDIO ---
AUDIO_COOLDOWN_WARN   = 5.0
AUDIO_COOLDOWN_CRIT   = 3.0

# --- DISPLAY ---
SHOW_LANDMARKS       = False
SHOW_POSE_AXES       = True
SHOW_EAR_VALUE       = True
SHOW_EULER_ANGLES    = True
SIDEBAR_WIDTH        = 320

# --- WARNA (BGR) ---
COLOR_OK        = (0,   200,   0)
COLOR_WARN      = (0,   200, 255)
COLOR_CRITICAL  = (0,    30, 220)
COLOR_OVERLAY   = (0,     0,   0)
COLOR_WHITE     = (255, 255, 255)
COLOR_GRAY      = (160, 160, 160)
COLOR_SIDEBAR   = (25,   25,  35)

# --- GAZE (IRIS) THRESHOLDS ---
# gaze_x: -1.0 = iris penuh ke kiri, +1.0 = iris penuh ke kanan
# Nilai ~0.0 = iris di tengah (lurus ke depan)
# Threshold 0.35 dipilih agar tidak terlalu sensitif terhadap noise iris
GAZE_LEFT_THRESHOLD  = -0.35   # gaze_x < nilai ini = melirik ke kiri
GAZE_RIGHT_THRESHOLD =  0.35   # gaze_x > nilai ini = melirik ke kanan

# Durasi threshold untuk eye gaze (lebih pendek dari head pose)
DURATION_WARN_GAZE   = 2.5    # detik sebelum WARNING
DURATION_CRIT_GAZE   = 5.0    # detik sebelum CRITICAL

# --- BEHAVIOR LABELS ---
BEHAVIOR_LABELS = {
    "LOOKING_LEFT":   "Looking Left",
    "LOOKING_RIGHT":  "Looking Right",
    "LOOKING_DOWN":   "Looking Down",
    "LOOKING_UP":     "Looking Up",
    "FACE_ABSENT":    "Face Absent",
    "EYES_CLOSED":    "Eyes Closed",
    "EYE_LOOK_LEFT":  "Eye Gaze Left",
    "EYE_LOOK_RIGHT": "Eye Gaze Right",
}