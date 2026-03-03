# script/06_replay_csv_drive.py
import os, sys, time
import pandas as pd
import carla
import keyboard

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.carla_utils import (
    connect,
    spawn_vehicle,
    sync_mode,
    follow_spectator_smooth,
    set_spectator_to_vehicle_once,
)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    client = connect()
    world = client.get_world()

    merged_path = os.path.join(PROJECT_ROOT, "data", "processed", "merged_all.csv")
    df = pd.read_csv(merged_path)

    # 只选一个 source_file（非常重要：不同文件可能 rot 约定不一致，但控制列通常一致）
    files = df["__source_file__"].unique().tolist()
    if not files:
        raise RuntimeError("No __source_file__ in merged_all.csv")

    # 默认选第一个文件；你也可以改成 files[k]
    chosen = files[0]
    g = df[df["__source_file__"] == chosen].copy()

    # 必需列
    need = ["time", "throttle", "steering"]
    miss = [c for c in need if c not in g.columns]
    if miss:
        raise RuntimeError(f"Missing columns in chosen file: {miss}")

    # 按时间排序
    g = g.sort_values("time").reset_index(drop=True)

    print("[INFO] chosen file:", chosen)
    print("[INFO] rows:", len(g))
    print("[INFO] Press ESC quit. Press SPACE to restart replay from beginning.")

    # 吞 WASD，避免相机乱跑（不吞 ESC/SPACE）
    for k in ["w", "a", "s", "d"]:
        keyboard.block_key(k)

    vehicle = None
    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3", spawn_index=0)
        cam_smooth = 0.10

        # 让脚本固定 20Hz 跑，跟你的数据节奏一致
        with sync_mode(world, fixed_delta_seconds=0.05):
            set_spectator_to_vehicle_once(world, vehicle)

            i = 0
            while True:
                if keyboard.is_pressed("esc"):
                    break
                if keyboard.is_pressed("space"):
                    i = 0
                    print("[INFO] restart replay")

                if i >= len(g):
                    # 播放完就停住
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                    world.tick()
                    continue

                row = g.iloc[i]
                thr = float(row["throttle"])
                steer = float(row["steering"])

                # 基础安全限幅（避免数据里偶发异常值）
                thr = clamp(thr, 0.0, 1.0)
                steer = clamp(steer, -1.0, 1.0)

                ctrl = carla.VehicleControl(
                    throttle=thr,
                    steer=steer,
                    brake=0.0,
                    reverse=False
                )
                vehicle.apply_control(ctrl)

                follow_spectator_smooth(
                    world, vehicle,
                    height=7.5, distance=9.5, pitch=-18.0,
                    smooth=cam_smooth
                )

                world.tick()
                i += 1

    finally:
        try:
            for k in ["w", "a", "s", "d"]:
                keyboard.unblock_key(k)
        except Exception:
            pass
        if vehicle is not None:
            vehicle.destroy()

if __name__ == "__main__":
    main()