from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

DEFAULT_HISTORY_STEPS = 20
DEFAULT_PREDICTION_STEPS = 15
DEFAULT_SAMPLE_RATE = 20.0

FEATURE_COLUMNS = [
    "dt",
    "throttle",
    "steering",
    "linear_speed",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "yaw_sin",
    "yaw_cos",
]

TARGET_COLUMNS = [
    "future_throttle",
    "future_steering",
    "future_dx_local",
    "future_dy_local",
    "future_dz",
    "future_rot6d_0",
    "future_rot6d_1",
    "future_rot6d_2",
    "future_rot6d_3",
    "future_rot6d_4",
    "future_rot6d_5",
]
