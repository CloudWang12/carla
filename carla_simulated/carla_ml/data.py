from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import FEATURE_COLUMNS, TARGET_COLUMNS

MAX_ABS_LINEAR_SPEED = 5.0
MAX_ABS_STEERING = 1.0
MAX_ABS_THROTTLE = 1.2
MIN_QUATERNION_NORM = 0.5
MAX_QUATERNION_NORM = 1.5
MIN_DT = 1e-3
MAX_DT = 0.2


def quaternion_to_yaw(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return quaternions / norms


def canonicalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    canonical = normalize_quaternions(quaternions.copy())
    sign = np.where(canonical[:, 3:4] < 0.0, -1.0, 1.0)
    canonical *= sign
    return canonical


def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = normalize_quaternions(quaternions)
    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    matrix = np.empty((len(quaternions), 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = ww + xx - yy - zz
    matrix[:, 0, 1] = 2.0 * (xy - zw)
    matrix[:, 0, 2] = 2.0 * (xz + yw)
    matrix[:, 1, 0] = 2.0 * (xy + zw)
    matrix[:, 1, 1] = ww - xx + yy - zz
    matrix[:, 1, 2] = 2.0 * (yz - xw)
    matrix[:, 2, 0] = 2.0 * (xz - yw)
    matrix[:, 2, 1] = 2.0 * (yz + xw)
    matrix[:, 2, 2] = ww - xx - yy + zz
    return matrix


def rotation_matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    first_two_columns = matrix[:, :, :2]
    return first_two_columns.reshape(len(matrix), 6).astype(np.float32)


def load_drive_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("time").reset_index(drop=True)

    required_numeric_columns = [
        "time",
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
    ]
    for column in required_numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    quaternion_norm = np.linalg.norm(df[["rot_0", "rot_1", "rot_2", "rot_3"]].to_numpy(dtype=np.float32), axis=1)
    finite_mask = np.isfinite(df[required_numeric_columns]).all(axis=1).to_numpy()
    valid_mask = (
        finite_mask
        & df["throttle"].abs().le(MAX_ABS_THROTTLE).to_numpy()
        & df["steering"].abs().le(MAX_ABS_STEERING).to_numpy()
        & df["linear_speed"].abs().le(MAX_ABS_LINEAR_SPEED).to_numpy()
        & (quaternion_norm >= MIN_QUATERNION_NORM)
        & (quaternion_norm <= MAX_QUATERNION_NORM)
    )
    if not bool(valid_mask.all()):
        df = df.loc[valid_mask].reset_index(drop=True)

    quaternions = canonicalize_quaternions(df[["rot_0", "rot_1", "rot_2", "rot_3"]].to_numpy(dtype=np.float32))
    df[["rot_0", "rot_1", "rot_2", "rot_3"]] = quaternions
    yaw = quaternion_to_yaw(
        df["rot_0"].to_numpy(),
        df["rot_1"].to_numpy(),
        df["rot_2"].to_numpy(),
        df["rot_3"].to_numpy(),
    )
    df["yaw"] = yaw
    df["yaw_sin"] = np.sin(yaw)
    df["yaw_cos"] = np.cos(yaw)
    dt = df["time"].diff().fillna(df["time"].diff().median())
    dt = dt.replace(0.0, dt[dt > 0].median() if (dt > 0).any() else 0.05)
    df["dt"] = dt.fillna(0.05).clip(lower=MIN_DT, upper=MAX_DT)
    return df


def compute_targets(df: pd.DataFrame, prediction_steps: int) -> np.ndarray:
    pos_x = df["pos_x"].to_numpy()
    pos_y = df["pos_y"].to_numpy()
    pos_z = df["pos_z"].to_numpy()
    yaw = df["yaw"].to_numpy()
    quaternions = df[["rot_0", "rot_1", "rot_2", "rot_3"]].to_numpy(dtype=np.float32)
    rot6d = rotation_matrix_to_rotation_6d(quaternion_to_rotation_matrix(quaternions))
    throttle = df["throttle"].to_numpy()
    steering = df["steering"].to_numpy()
    length = len(df)

    targets = np.zeros((length, prediction_steps, len(TARGET_COLUMNS)), dtype=np.float32)
    for idx in range(length):
        base_x = pos_x[idx]
        base_y = pos_y[idx]
        base_z = pos_z[idx]
        base_yaw = yaw[idx]
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        for step in range(prediction_steps):
            future_idx = min(idx + step + 1, length - 1)
            dx_world = pos_x[future_idx] - base_x
            dy_world = pos_y[future_idx] - base_y
            dx_local = cos_yaw * dx_world + sin_yaw * dy_world
            dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
            dz = pos_z[future_idx] - base_z
            targets[idx, step] = np.array(
                [
                    throttle[future_idx],
                    steering[future_idx],
                    dx_local,
                    dy_local,
                    dz,
                    rot6d[future_idx, 0],
                    rot6d[future_idx, 1],
                    rot6d[future_idx, 2],
                    rot6d[future_idx, 3],
                    rot6d[future_idx, 4],
                    rot6d[future_idx, 5],
                ],
                dtype=np.float32,
            )
    return targets


@dataclass
class SequenceBundle:
    inputs: np.ndarray
    targets: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def save_metadata(self, path: Path) -> None:
        payload = {
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SequenceDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def build_sequences(
    dataset_dir: Path,
    history_steps: int,
    prediction_steps: int,
    validation_ratio: float,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, SequenceBundle]:
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for csv_path in sorted(dataset_dir.rglob("*.csv")):
        df = load_drive_csv(csv_path)
        if len(df) < history_steps + prediction_steps:
            continue
        features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = compute_targets(df, prediction_steps)
        max_start = len(df) - history_steps - prediction_steps + 1
        for start in range(max_start):
            end = start + history_steps
            all_inputs.append(features[start:end])
            all_targets.append(targets[end - 1])

    if not all_inputs:
        raise ValueError(f"No usable training sequences found under {dataset_dir}")

    inputs = np.stack(all_inputs)
    targets = np.stack(all_targets)

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs,
        targets,
        test_size=validation_ratio,
        random_state=42,
        shuffle=True,
    )

    feature_mean = train_inputs.mean(axis=(0, 1))
    feature_std = train_inputs.std(axis=(0, 1))
    feature_std[feature_std < 1e-6] = 1.0

    target_mean = train_targets.mean(axis=(0, 1))
    target_std = train_targets.std(axis=(0, 1))
    target_std[target_std < 1e-6] = 1.0

    train_inputs = (train_inputs - feature_mean) / feature_std
    val_inputs = (val_inputs - feature_mean) / feature_std
    train_targets = (train_targets - target_mean) / target_std
    val_targets = (val_targets - target_mean) / target_std

    train_loader = DataLoader(
        SequenceDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        SequenceDataset(val_inputs, val_targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    bundle = SequenceBundle(
        inputs=inputs,
        targets=targets,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
    )
    return train_loader, val_loader, bundle
