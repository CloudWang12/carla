# script/04_run_bc_policy_in_carla.py
import os, sys, json, time
from collections import deque

import numpy as np
import torch
import carla
import keyboard  # pip install keyboard

# --- sys.path 注入，保证能 import src ---
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
from src.feature_runtime import build_feature_vector, speed_mps


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_policy():
    model_path = os.path.join(PROJECT_ROOT, "models", "bc_policy.pt")
    meta_path = os.path.join(PROJECT_ROOT, "models", "bc_policy_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise RuntimeError("Missing models/bc_policy.pt or models/bc_policy_meta.json (run training first).")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    label_cols = meta["label_cols"]
    window = int(meta["window"])

    x_mu = np.array(meta["x_mu"], dtype=np.float32)
    x_sd = np.array(meta["x_sd"], dtype=np.float32)
    y_mu = np.array(meta["y_mu"], dtype=np.float32)
    y_sd = np.array(meta["y_sd"], dtype=np.float32)

    in_dim = len(feature_cols) * window
    model = MLPPolicy(in_dim=in_dim, out_dim=len(label_cols))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, window, x_mu, x_sd, y_mu, y_sd


def main():
    client = connect()
    world = client.get_world()

    # ====== 吞键：彻底消灭 spectator 相机偏移 ======
    # 不吞 T（切换模式）
    for k in ["w", "a", "s", "d", "space", "esc"]:
        keyboard.block_key(k)

    vehicle = None
    imu = None
    latest_imu = None

    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3", spawn_index=0)

        # 加载模型
        model, WINDOW, x_mu, x_sd, y_mu, y_sd = load_policy()
        F = len(x_mu) // WINDOW  # 单帧特征维度（按 meta 自动推断）
        print("[OK] Policy loaded. WINDOW =", WINDOW, "F =", F)

        # 绑定 IMU
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.05")  # 与 fixed_delta=0.05 对齐（20Hz）
        imu_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        imu = world.spawn_actor(imu_bp, imu_tf, attach_to=vehicle)

        def on_imu(meas):
            nonlocal latest_imu
            latest_imu = meas

        imu.listen(on_imu)

        # 滑窗队列：每帧一个 feature vector（长度 F）
        feat_buf = deque(maxlen=WINDOW)

        # 控制模式：手动/自动
        auto_mode = False
        last_toggle = 0.0

        # 手动控制（慢速参数）
        control = carla.VehicleControl()
        THROTTLE_UP = 0.35
        BRAKE_UP = 0.65
        STEER_UP = 1.40
        THROTTLE_DOWN = 0.45
        BRAKE_DOWN = 0.90
        STEER_RETURN = 2.20

        # 倒车阈值
        STOP_SPEED = 0.30
        REVERSE_THROTTLE_MAX = 0.45

        cam_smooth = 0.10

        print("[INFO] Press T to toggle AUTO/MANUAL.")
        print("[INFO] MANUAL: hold W/S/A/D, SPACE reset, ESC quit.")
        print("[INFO] AUTO: model outputs steering/throttle (forward).")

        # ====== 关键：把仿真 tick 对齐到你的数据（建议 0.05=20Hz）=====
        with sync_mode(world, fixed_delta_seconds=0.05):
            set_spectator_to_vehicle_once(world, vehicle)

            last = time.time()
            step_i = 0

            while True:
                now = time.time()
                dt = clamp(now - last, 0.0, 0.2)  # 20Hz 下 dt ~ 0.05
                last = now

                # 退出
                if keyboard.is_pressed("esc"):
                    break

                # 切换模式（防抖：0.3s）
                if keyboard.is_pressed("t") and (now - last_toggle) > 0.3:
                    auto_mode = not auto_mode
                    last_toggle = now
                    print("[MODE]", "AUTO" if auto_mode else "MANUAL")
                    # 切换时先清一下控制，避免残留
                    control = carla.VehicleControl()
                    feat_buf.clear()
                    # 清平滑状态
                    if hasattr(main, "_prev_steer"):
                        delattr(main, "_prev_steer")
                    if hasattr(main, "_prev_thr"):
                        delattr(main, "_prev_thr")

                # reset
                if keyboard.is_pressed("space"):
                    control = carla.VehicleControl()
                    feat_buf.clear()
                    if hasattr(main, "_prev_steer"):
                        delattr(main, "_prev_steer")
                    if hasattr(main, "_prev_thr"):
                        delattr(main, "_prev_thr")

                if not auto_mode:
                    # ================== 手动模式（慢速版）==================
                    spd = speed_mps(vehicle)

                    w = keyboard.is_pressed("w")
                    s = keyboard.is_pressed("s")
                    a = keyboard.is_pressed("a")
                    d = keyboard.is_pressed("d")

                    # steer
                    if a and not d:
                        control.steer = clamp(control.steer - STEER_UP * dt, -1.0, 1.0)
                    elif d and not a:
                        control.steer = clamp(control.steer + STEER_UP * dt, -1.0, 1.0)
                    else:
                        if control.steer > 0:
                            control.steer = max(0.0, control.steer - STEER_RETURN * dt)
                        elif control.steer < 0:
                            control.steer = min(0.0, control.steer + STEER_RETURN * dt)

                    # throttle/brake/reverse
                    if w and not s:
                        control.reverse = False
                        control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)
                        control.throttle = clamp(control.throttle + THROTTLE_UP * dt, 0.0, 1.0)
                    elif s and not w:
                        if spd > STOP_SPEED and not control.reverse:
                            control.throttle = max(0.0, control.throttle - THROTTLE_DOWN * dt)
                            control.brake = clamp(control.brake + BRAKE_UP * dt, 0.0, 1.0)
                            control.reverse = False
                        else:
                            control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)
                            control.reverse = True
                            control.throttle = clamp(
                                control.throttle + THROTTLE_UP * dt, 0.0, REVERSE_THROTTLE_MAX
                            )
                    else:
                        control.throttle = max(0.0, control.throttle - THROTTLE_DOWN * dt)
                        control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)

                else:
                    # ================== 自动模式：模型推理输出 steering/throttle ==================
                    feat = build_feature_vector(vehicle, latest_imu)

                    # 安全：确保维度匹配
                    if len(feat) != F:
                        raise RuntimeError(f"Feature dim mismatch. got {len(feat)} expected {F}")

                    feat_buf.append(feat)

                    if len(feat_buf) < WINDOW:
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, reverse=False)
                    else:
                        x = np.array(feat_buf, dtype=np.float32).reshape(-1)  # [WINDOW*F]
                        x = (x - x_mu) / x_sd

                        with torch.no_grad():
                            y_norm = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()

                        y = y_norm * y_sd + y_mu

                        steer = float(y[0])
                        throttle = float(y[1])

                        # ======= 输出后处理：稳态化（抑制左右抖动 + 提速）=======
                        step_i += 1

                        # 1) 基础限幅
                        steer = clamp(steer, -0.6, 0.6)
                        throttle = clamp(throttle, 0.0, 1.0)

                        # 2) 死区抑制 0 附近抖动
                        deadband = 0.06
                        if abs(steer) < deadband:
                            steer = 0.0

                        # 3) 低通平滑 + 速率限制（核心）
                        alpha = 0.12
                        prev_steer = getattr(main, "_prev_steer", 0.0)
                        steer_lp = prev_steer + (steer - prev_steer) * alpha

                        max_steer_rate = 0.9  # per second
                        max_delta = max_steer_rate * dt
                        steer_final = clamp(steer_lp, prev_steer - max_delta, prev_steer + max_delta)
                        main._prev_steer = steer_final

                        # 4) 速度控制：目标速度 + 油门兜底（避免“慢慢挪”）
                        target_speed = 6.0  # m/s
                        cur_speed = speed_mps(vehicle)
                        k_p = 0.12
                        throttle_pid = clamp(k_p * (target_speed - cur_speed), 0.0, 0.6)

                        throttle_final = max(throttle, throttle_pid, 0.18)
                        throttle_final = clamp(throttle_final, 0.0, 0.7)

                        # 可选：对油门也做一点平滑（防抖）
                        prev_thr = getattr(main, "_prev_thr", throttle_final)
                        thr_alpha = 0.15
                        throttle_final = prev_thr + (throttle_final - prev_thr) * thr_alpha
                        main._prev_thr = throttle_final

                        # 调试打印（每 30 帧）
                        if step_i % 30 == 0:
                            print(f"[AUTO] steer={steer_final:+.3f} thr={throttle_final:.3f} speed={cur_speed:.2f}")

                        control = carla.VehicleControl(
                            throttle=float(throttle_final),
                            steer=float(steer_final),
                            brake=0.0,
                            reverse=False
                        )

                # 应用控制
                vehicle.apply_control(control)

                # 锁相机
                follow_spectator_smooth(
                    world, vehicle,
                    height=7.5, distance=9.5, pitch=-18.0,
                    smooth=cam_smooth
                )

                # tick
                world.tick()

    finally:
        # 解除吞键
        try:
            for k in ["w", "a", "s", "d", "space", "esc"]:
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