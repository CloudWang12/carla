# script/04_run_groups_policy_in_carla.py
# 功能：
# - T：MANUAL / AUTO（模型）切换
# - G：循环切换 group（14个）并自动加载对应模型
# - MANUAL：WASD（慢速+倒车），手感优化（油门/转向限幅+更自然回落）
# - AUTO：模型输出 steering/throttle + 稳态化（转向滤波+速率限制、油门PID+斜率限制），手感优化
# - 固定 20Hz（fixed_delta_seconds=0.05），相机跟随且不会被 WASD 影响（吞键）
# - 全局硬限速：超过 MAX_SPEED 强制断油+轻刹（MANUAL/AUTO 都生效）

import os, sys, json, time
from collections import deque

import numpy as np
import torch
import carla
import keyboard  # pip install keyboard

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
from src.model_bc import MLPPolicy


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)


def rot_to_quat_wxyz(rot: carla.Rotation):
    import math
    roll = math.radians(rot.roll)
    pitch = math.radians(rot.pitch)
    yaw = math.radians(rot.yaw)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def discover_groups_models():
    base = os.path.join(PROJECT_ROOT, "models", "groups")
    if not os.path.isdir(base):
        raise RuntimeError("No models/groups found. Run script/03_train_groups.py first.")
    groups = []
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "policy.pt")) and os.path.exists(os.path.join(d, "meta.json")):
            groups.append(name)
    groups.sort()
    if not groups:
        raise RuntimeError("models/groups exists but no valid group models found.")
    return groups


def load_group_model(group: str):
    gdir = os.path.join(PROJECT_ROOT, "models", "groups", group)
    meta_path = os.path.join(gdir, "meta.json")
    pt_path = os.path.join(gdir, "policy.pt")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    window = int(meta["window"])
    feature_cols = meta["feature_cols"]
    label_cols = meta["label_cols"]
    x_mu = np.array(meta["x_mu"], dtype=np.float32)
    x_sd = np.array(meta["x_sd"], dtype=np.float32)
    y_mu = np.array(meta["y_mu"], dtype=np.float32)
    y_sd = np.array(meta["y_sd"], dtype=np.float32)

    in_dim = len(feature_cols) * window
    model = MLPPolicy(in_dim=in_dim, out_dim=len(label_cols))
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    return model, meta, window, feature_cols, x_mu, x_sd, y_mu, y_sd


def build_feature_from_meta(vehicle: carla.Vehicle, imu_meas, feature_cols: list[str]):
    """
    支持的字段：
    linear_speed, gyro_*, acc_*, rot_0..3 (xyzw), pos_*
    训练时 meta.feature_cols 决定实际使用哪些。
    """
    tf = vehicle.get_transform()
    loc = tf.location
    rot = tf.rotation
    spd = speed_mps(vehicle)

    if imu_meas is None:
        gx = gy = gz = 0.0
        ax = ay = az = 0.0
    else:
        gx, gy, gz = float(imu_meas.gyroscope.x), float(imu_meas.gyroscope.y), float(imu_meas.gyroscope.z)
        ax, ay, az = float(imu_meas.accelerometer.x), float(imu_meas.accelerometer.y), float(imu_meas.accelerometer.z)

    qw, qx, qy, qz = rot_to_quat_wxyz(rot)
    # 在线统一输出 rot_0..3 = x,y,z,w（xyzw）
    q_xyzw = (qx, qy, qz, qw)

    m = {
        "linear_speed": spd,
        "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
        "acc_x": ax, "acc_y": ay, "acc_z": az,
        "rot_0": q_xyzw[0], "rot_1": q_xyzw[1], "rot_2": q_xyzw[2], "rot_3": q_xyzw[3],
        "pos_x": float(loc.x), "pos_y": float(loc.y), "pos_z": float(loc.z),
    }
    return [float(m[c]) for c in feature_cols]


def main():
    client = connect()
    world = client.get_world()

    # 吞 WASD，防止 spectator 相机被键盘影响（不吞 T/G/ESC/SPACE）
    for k in ["w", "a", "s", "d"]:
        keyboard.block_key(k)

    vehicle = None
    imu = None
    latest_imu = None

    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3", spawn_index=0)

        groups = discover_groups_models()
        gi = 0
        model, meta, WINDOW, feature_cols, x_mu, x_sd, y_mu, y_sd = load_group_model(groups[gi])
        feat_buf = deque(maxlen=WINDOW)

        print("[INFO] groups:", len(groups))
        print("[INFO] current group:", groups[gi])
        print("[INFO] Press G to switch group, T to toggle MANUAL/AUTO.")
        print("[INFO] MANUAL: hold WASD, SPACE reset, ESC quit.")
        print("[INFO] AUTO: model outputs steering/throttle for current group.")

        # IMU
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.05")
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

        def on_imu(meas):
            nonlocal latest_imu
            latest_imu = meas

        imu.listen(on_imu)

        auto_mode = False
        last_toggle = 0.0
        last_group_switch = 0.0

        # -------------------------
        # 全局硬限速（MANUAL/AUTO 都生效）
        # -------------------------
        MAX_SPEED = 3.5      # m/s ≈ 12.6 km/h
        SPEED_BRAKE = 0.25   # 超速时最小刹车

        # -------------------------
        # MANUAL 手感参数（标准车感 / 20Hz）
        # -------------------------
        THROTTLE_UP = 0.55
        THROTTLE_DOWN = 0.85

        BRAKE_UP = 1.10
        BRAKE_DOWN = 1.40

        STEER_UP = 2.20
        STEER_RETURN = 3.20

        MAX_MANUAL_THROTTLE = 0.55
        MAX_MANUAL_BRAKE = 1.00
        MAX_MANUAL_STEER = 0.65

        # 倒车
        STOP_SPEED = 0.30
        REVERSE_THROTTLE_MAX = 0.35

        # -------------------------
        # AUTO 手感参数（稳态化 + 限速更自然）
        # -------------------------
        deadband = 0.08
        steer_alpha = 0.08
        max_steer_rate = 0.5  # per second

        # 目标速度（更慢）
        target_speed = 3.0
        k_p = 0.18

        THR_MIN = 0.12
        THR_MAX = 0.45
        THR_SLEW = 0.35  # 每秒油门最大变化（关键：不憋、不猛）

        # 相机
        cam_smooth = 0.10

        with sync_mode(world, fixed_delta_seconds=0.05):
            set_spectator_to_vehicle_once(world, vehicle)

            last = time.time()
            step_i = 0

            while True:
                now = time.time()
                dt = clamp(now - last, 0.0, 0.2)
                last = now

                if keyboard.is_pressed("esc"):
                    break

                # T 切换模式
                if keyboard.is_pressed("t") and (now - last_toggle) > 0.3:
                    auto_mode = not auto_mode
                    last_toggle = now
                    print("[MODE]", "AUTO" if auto_mode else "MANUAL")
                    feat_buf.clear()
                    for a in ["_prev_steer", "_prev_thr"]:
                        if hasattr(main, a):
                            delattr(main, a)

                # G 切换 group
                if keyboard.is_pressed("g") and (now - last_group_switch) > 0.3:
                    last_group_switch = now
                    gi = (gi + 1) % len(groups)
                    model, meta, WINDOW, feature_cols, x_mu, x_sd, y_mu, y_sd = load_group_model(groups[gi])
                    feat_buf = deque(maxlen=WINDOW)
                    for a in ["_prev_steer", "_prev_thr"]:
                        if hasattr(main, a):
                            delattr(main, a)
                    print("[GROUP] switched to:", groups[gi], "| feature_dim:", len(feature_cols))

                # SPACE reset
                if keyboard.is_pressed("space"):
                    feat_buf.clear()
                    for a in ["_prev_steer", "_prev_thr"]:
                        if hasattr(main, a):
                            delattr(main, a)

                # 先生成 control（后面会统一经过“硬限速层”）
                if not auto_mode:
                    # ---------------- MANUAL ----------------
                    control = carla.VehicleControl()
                    spd = speed_mps(vehicle)

                    w = keyboard.is_pressed("w")
                    s = keyboard.is_pressed("s")
                    a = keyboard.is_pressed("a")
                    d = keyboard.is_pressed("d")

                    # steer（限幅到 MAX_MANUAL_STEER）
                    prev_steer = getattr(main, "_manual_steer", 0.0)
                    steer = prev_steer
                    if a and not d:
                        steer = clamp(steer - STEER_UP * dt, -MAX_MANUAL_STEER, MAX_MANUAL_STEER)
                    elif d and not a:
                        steer = clamp(steer + STEER_UP * dt, -MAX_MANUAL_STEER, MAX_MANUAL_STEER)
                    else:
                        if steer > 0:
                            steer = max(0.0, steer - STEER_RETURN * dt)
                        elif steer < 0:
                            steer = min(0.0, steer + STEER_RETURN * dt)
                    main._manual_steer = steer

                    # throttle/brake/reverse（两段式 S：先刹停再倒车）
                    prev_thr = getattr(main, "_manual_thr", 0.0)
                    prev_brk = getattr(main, "_manual_brk", 0.0)
                    thr, brk = prev_thr, prev_brk
                    rev = getattr(main, "_manual_rev", False)

                    if w and not s:
                        rev = False
                        brk = max(0.0, brk - BRAKE_DOWN * dt)
                        thr = clamp(thr + THROTTLE_UP * dt, 0.0, MAX_MANUAL_THROTTLE)

                    elif s and not w:
                        if spd > STOP_SPEED and not rev:
                            thr = max(0.0, thr - THROTTLE_DOWN * dt)
                            brk = clamp(brk + BRAKE_UP * dt, 0.0, MAX_MANUAL_BRAKE)
                            rev = False
                        else:
                            brk = max(0.0, brk - BRAKE_DOWN * dt)
                            rev = True
                            thr = clamp(thr + THROTTLE_UP * dt, 0.0, REVERSE_THROTTLE_MAX)

                    else:
                        thr = max(0.0, thr - THROTTLE_DOWN * dt)
                        brk = max(0.0, brk - BRAKE_DOWN * dt)

                    main._manual_thr = thr
                    main._manual_brk = brk
                    main._manual_rev = rev

                    control.throttle = float(thr)
                    control.brake = float(brk)
                    control.steer = float(steer)
                    control.reverse = bool(rev)

                else:
                    # ---------------- AUTO (MODEL) ----------------
                    # 缓冲未满：先积累
                    feat = build_feature_from_meta(vehicle, latest_imu, feature_cols)
                    feat_buf.append(feat)

                    if len(feat_buf) < WINDOW:
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, reverse=False)
                    else:
                        x = np.array(feat_buf, dtype=np.float32).reshape(-1)
                        x = (x - x_mu) / x_sd

                        with torch.no_grad():
                            y_norm = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
                        y = y_norm * y_sd + y_mu

                        steer = float(y[0])
                        thr_model = float(y[1])

                        # ---- 转向稳态化：死区 + 平滑 + 速率限制 ----
                        steer = clamp(steer, -0.8, 0.8)
                        if abs(steer) < deadband:
                            steer = 0.0

                        prev_steer = getattr(main, "_prev_steer", 0.0)
                        steer_lp = prev_steer + (steer - prev_steer) * steer_alpha
                        max_delta = max_steer_rate * dt
                        steer_final = clamp(steer_lp, prev_steer - max_delta, prev_steer + max_delta)
                        main._prev_steer = steer_final

                        # ---- 速度控制 + 油门斜率限制 ----
                        cur_speed = speed_mps(vehicle)
                        thr_pid = clamp(k_p * (target_speed - cur_speed), 0.0, 0.6)

                        thr_cmd = max(thr_model, thr_pid, THR_MIN)
                        thr_cmd = clamp(thr_cmd, 0.0, THR_MAX)

                        prev_thr = getattr(main, "_prev_thr", 0.0)
                        max_thr_delta = THR_SLEW * dt
                        thr_limited = clamp(thr_cmd, prev_thr - max_thr_delta, prev_thr + max_thr_delta)
                        throttle_final = clamp(max(thr_limited, THR_MIN), 0.0, THR_MAX)
                        main._prev_thr = throttle_final

                        step_i += 1
                        if step_i % 30 == 0:
                            print(f"[AUTO:{groups[gi]}] steer={steer_final:+.3f} thr={throttle_final:.3f} speed={cur_speed:.2f}")

                        control = carla.VehicleControl(
                            throttle=float(throttle_final),
                            steer=float(steer_final),
                            brake=0.0,
                            reverse=False
                        )

                # ---------------- 全局硬限速层（最关键） ----------------
                cur_speed = speed_mps(vehicle)
                if cur_speed > MAX_SPEED:
                    control.throttle = 0.0
                    control.brake = max(control.brake, SPEED_BRAKE)

                vehicle.apply_control(control)
                follow_spectator_smooth(world, vehicle, height=7.5, distance=9.5, pitch=-18.0, smooth=cam_smooth)
                world.tick()

    finally:
        try:
            for k in ["w", "a", "s", "d"]:
                keyboard.unblock_key(k)
        except Exception:
            pass

        if imu is not None:
            try:
                imu.stop()
            except Exception:
                pass
            imu.destroy()

        if vehicle is not None:
            vehicle.destroy()


if __name__ == "__main__":
    main()