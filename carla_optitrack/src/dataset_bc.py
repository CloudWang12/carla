# src/dataset_bc.py
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "time",
    "throttle", "steering",
    "linear_speed",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z",
    "pos_x", "pos_y", "pos_z",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "voltage",
]

def find_csv_files(root_dir: str) -> list[str]:
    pattern = os.path.join(root_dir, "**", "*.csv")
    return sorted(glob.glob(pattern, recursive=True))

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    df["__source_file__"] = path
    return df

def merge_all_csv(root_dir: str) -> pd.DataFrame:
    paths = find_csv_files(root_dir)
    if not paths:
        raise RuntimeError(f"No csv under {root_dir}")

    dfs = []
    for p in paths:
        dfs.append(load_and_validate(p))
    out = pd.concat(dfs, ignore_index=True)

    # 可选：按 file 内 time 排序，防止乱序
    out.sort_values(["__source_file__", "time"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    每个 file 内做滑窗：X shape = [N, window * F], Y shape = [N, L]
    Y 默认取窗口最后一帧的控制标签
    """
    Xs, Ys = [], []
    for _, g in df.groupby("__source_file__", sort=False):
        g = g.dropna(subset=feature_cols + label_cols)
        if len(g) < window:
            continue
        feat = g[feature_cols].to_numpy(np.float32)
        lab = g[label_cols].to_numpy(np.float32)

        for i in range(window - 1, len(g)):
            x_win = feat[i - window + 1 : i + 1].reshape(-1)
            y = lab[i]
            Xs.append(x_win)
            Ys.append(y)

    if not Xs:
        raise RuntimeError("No usable windows built. Check data/columns.")
    return np.stack(Xs, axis=0), np.stack(Ys, axis=0)