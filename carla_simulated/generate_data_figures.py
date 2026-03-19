from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from carla_ml.config import FEATURE_COLUMNS
from carla_ml.data import canonicalize_quaternions, load_drive_csv, normalize_quaternions
from carla_ml.model import DrivingLSTM


PLOT_COLUMNS = ["pos_x", "pos_y", "pos_z", "yaw", "rot_0", "rot_1", "rot_2", "rot_3"]


@dataclass
class ModelArtifacts:
    model: DrivingLSTM
    device: torch.device
    history_steps: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def align_quaternions(reference: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    aligned = quaternions.copy()
    dot = np.sum(reference * aligned, axis=1, keepdims=True)
    signs = np.where(dot < 0.0, -1.0, 1.0).astype(np.float32)
    return aligned * signs


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    a1 = rotation_6d[:3]
    a2 = rotation_6d[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    quat = normalize_quaternions(quat[None, :])[0]
    if quat[3] < 0.0:
        quat = -quat
    return quat


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def load_model(model_path: Path, metadata_path: Path, device_name: str) -> ModelArtifacts:
    checkpoint = torch.load(model_path, map_location=device_name)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = DrivingLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        prediction_steps=checkpoint["prediction_steps"],
        target_size=checkpoint["target_size"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return ModelArtifacts(
        model=model,
        device=device,
        history_steps=metadata["history_steps"],
        feature_mean=np.array(metadata["feature_mean"], dtype=np.float32),
        feature_std=np.array(metadata["feature_std"], dtype=np.float32),
        target_mean=np.array(metadata["target_mean"], dtype=np.float32),
        target_std=np.array(metadata["target_std"], dtype=np.float32),
    )


def predict_next_step(history: np.ndarray, artifacts: ModelArtifacts) -> np.ndarray:
    normalized = (history - artifacts.feature_mean) / artifacts.feature_std
    tensor = torch.tensor(normalized[None, ...], dtype=torch.float32, device=artifacts.device)
    with torch.no_grad():
        prediction = artifacts.model(tensor).cpu().numpy()[0]
    return prediction[0] * artifacts.target_std + artifacts.target_mean


def reconstruct_predicted_state(row: np.ndarray, predicted_step: np.ndarray) -> np.ndarray:
    base_x = float(row[0])
    base_y = float(row[1])
    base_z = float(row[2])
    base_yaw = float(row[3])
    dx_local = float(predicted_step[2])
    dy_local = float(predicted_step[3])
    dz = float(predicted_step[4])

    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    dx_world = cos_yaw * dx_local - sin_yaw * dy_local
    dy_world = sin_yaw * dx_local + cos_yaw * dy_local
    predicted_quat = rotation_matrix_to_quaternion(rotation_6d_to_matrix(predicted_step[5:11]))
    _, _, next_yaw = quaternion_to_euler(
        float(predicted_quat[0]),
        float(predicted_quat[1]),
        float(predicted_quat[2]),
        float(predicted_quat[3]),
    )
    return np.array(
        [
            base_x + dx_world,
            base_y + dy_world,
            base_z + dz,
            next_yaw,
            predicted_quat[0],
            predicted_quat[1],
            predicted_quat[2],
            predicted_quat[3],
        ],
        dtype=np.float32,
    )


def build_run_series(csv_path: Path, artifacts: ModelArtifacts) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    df = load_drive_csv(csv_path)
    if len(df) <= artifacts.history_steps:
        return None

    feature_array = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    state_array = df[PLOT_COLUMNS].to_numpy(dtype=np.float32)
    times = df["time"].to_numpy(dtype=np.float64)

    raw_times: list[float] = []
    raw_states: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []

    for end_idx in range(artifacts.history_steps - 1, len(df) - 1):
        history = feature_array[end_idx - artifacts.history_steps + 1 : end_idx + 1]
        predicted_step = predict_next_step(history, artifacts)
        predicted_state = reconstruct_predicted_state(state_array[end_idx], predicted_step)

        future_idx = end_idx + 1
        raw_times.append(float(times[future_idx] - times[artifacts.history_steps]))
        raw_states.append(state_array[future_idx])
        predicted_states.append(predicted_state)

    if not raw_times:
        return None

    raw_series = np.stack(raw_states)
    predicted_series = np.stack(predicted_states)
    raw_series[:, 4:8] = canonicalize_quaternions(raw_series[:, 4:8])
    predicted_series[:, 4:8] = canonicalize_quaternions(predicted_series[:, 4:8])
    predicted_series[:, 4:8] = align_quaternions(raw_series[:, 4:8], predicted_series[:, 4:8])

    baseline_pos = raw_series[0, :3].copy()
    raw_series[:, :3] = raw_series[:, :3] - baseline_pos
    predicted_series[:, :3] = predicted_series[:, :3] - baseline_pos
    baseline_yaw = raw_series[0, 3]
    raw_series[:, 3] = np.unwrap(raw_series[:, 3] - baseline_yaw)
    predicted_series[:, 3] = np.unwrap(predicted_series[:, 3] - baseline_yaw)
    return np.array(raw_times, dtype=np.float64), raw_series, predicted_series


def interpolate_series(time_axis: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    interpolated = np.empty((len(target_time), values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        interpolated[:, dim] = np.interp(target_time, time_axis, values[:, dim]).astype(np.float32)
    return interpolated


def summarize_logic(logic_dir: Path, artifacts: ModelArtifacts) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for csv_path in sorted(logic_dir.glob("*.csv")):
        run = build_run_series(csv_path, artifacts)
        if run is not None:
            runs.append(run)

    if not runs:
        return None

    common_end = min(run[0][-1] for run in runs if len(run[0]) > 1)
    if common_end <= 0.0:
        return None

    sample_count = max(200, min(800, int(max(len(run[0]) for run in runs))))
    target_time = np.linspace(0.0, common_end, sample_count, dtype=np.float64)

    raw_stack = []
    pred_stack = []
    for time_axis, raw_values, pred_values in runs:
        raw_stack.append(interpolate_series(time_axis, raw_values, target_time))
        pred_stack.append(interpolate_series(time_axis, pred_values, target_time))

    mean_raw = np.mean(np.stack(raw_stack), axis=0)
    mean_pred = np.mean(np.stack(pred_stack), axis=0)
    return target_time, mean_raw, mean_pred


def plot_logic(logic_name: str, target_time: np.ndarray, raw_values: np.ndarray, pred_values: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
    axes = axes.flatten()

    for idx, column in enumerate(PLOT_COLUMNS):
        ax = axes[idx]
        ax.plot(target_time, raw_values[:, idx], label="raw", linewidth=2.0, color="#1f77b4")
        ax.plot(target_time, pred_values[:, idx], label="model", linewidth=1.8, color="#d62728", alpha=0.9)
        ax.set_title(column)
        ax.set_ylabel("delta" if idx < 4 else "value")
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper left")
    for ax in axes:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{logic_name}: raw vs model-reconstructed trajectory", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw-vs-model trajectory figures for each dataset logic.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/driving_lstm.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("artifacts/models/driving_lstm_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_figure"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_model(args.model_path, args.metadata_path, args.device)

    created = 0
    for logic_dir in sorted(path for path in args.dataset_dir.iterdir() if path.is_dir()):
        summary = summarize_logic(logic_dir, artifacts)
        if summary is None:
            continue
        target_time, raw_values, pred_values = summary
        output_path = args.output_dir / f"{logic_dir.name}.png"
        plot_logic(logic_dir.name, target_time, raw_values, pred_values, output_path)
        created += 1
        print(f"saved {output_path}")

    print(f"generated {created} figure(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
