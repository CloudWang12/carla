# scripts/01_keyboard_drive_suppress.py
import time
import carla
import keyboard  # pip install keyboard

from src.carla_utils import (
    connect,
    spawn_vehicle,
    sync_mode,
    follow_spectator_smooth,
    set_spectator_to_vehicle_once,
)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5

def main():
    client = connect()
    world = client.get_world()

    vehicle = None
    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3", spawn_index=0)

        print("[INFO] keyboard(suppress) ON: WASD won't reach CARLA window.")
        print("[INFO] Hold W: accelerate forward")
        print("[INFO] Hold S: brake to stop, then reverse")
        print("[INFO] Hold A/D: steer")
        print("[INFO] SPACE: reset | ESC: quit")
        print("[TIP] If suppress doesn't work, run Python/PyCharm as Administrator.")

        # 吞键：彻底阻止 CARLA spectator 被 WASD 控制
        for k in ["w", "a", "s", "d", "space", "esc"]:
            keyboard.block_key(k)

        # —— 手感参数：你说太快，就把这些调小 ——
        # “上升速度”（每秒变化量）
        THROTTLE_UP = 0.1     # 油门上升更慢（原来 0.70）
        BRAKE_UP = 0.65        # 刹车上升更慢（原来 1.20）
        STEER_UP = 0.5         # 转向变化更慢（原来 2.40）

        # “回落速度”（松手时回到 0 的速度）
        THROTTLE_DOWN = 0.45
        BRAKE_DOWN = 0.90
        STEER_RETURN = 2.20

        # 倒车逻辑阈值
        STOP_SPEED = 0.30      # m/s，低于这个认为基本停住，可进入倒车
        REVERSE_THROTTLE_MAX = 0.45  # 倒车最大油门（避免倒车太冲）

        cam_smooth = 0.10

        with sync_mode(world, fixed_delta_seconds=1.0 / 60.0):
            set_spectator_to_vehicle_once(world, vehicle)

            control = carla.VehicleControl()
            last = time.time()

            while True:
                now = time.time()
                dt = clamp(now - last, 0.0, 0.05)
                last = now

                # 退出
                if keyboard.is_pressed("esc"):
                    break

                # 重置
                if keyboard.is_pressed("space"):
                    control = carla.VehicleControl()

                # 当前速度
                spd = speed_mps(vehicle)

                w = keyboard.is_pressed("w")
                s = keyboard.is_pressed("s")
                a = keyboard.is_pressed("a")
                d = keyboard.is_pressed("d")

                # --------- 方向（按住式 + 松手回正）---------
                if a and not d:
                    control.steer = clamp(control.steer - STEER_UP * dt, -1.0, 1.0)
                elif d and not a:
                    control.steer = clamp(control.steer + STEER_UP * dt, -1.0, 1.0)
                else:
                    # 回正
                    if control.steer > 0:
                        control.steer = max(0.0, control.steer - STEER_RETURN * dt)
                    elif control.steer < 0:
                        control.steer = min(0.0, control.steer + STEER_RETURN * dt)

                # --------- 前进 / 刹车 / 倒车（两段式S）---------
                if w and not s:
                    # 前进：取消倒车，逐渐加油
                    control.reverse = False
                    control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)
                    control.throttle = clamp(control.throttle + THROTTLE_UP * dt, 0.0, 1.0)

                elif s and not w:
                    # S：先刹到接近 0，再倒车
                    if spd > STOP_SPEED and not control.reverse:
                        # 阶段1：刹停
                        control.throttle = max(0.0, control.throttle - THROTTLE_DOWN * dt)
                        control.brake = clamp(control.brake + BRAKE_UP * dt, 0.0, 1.0)
                        control.reverse = False
                    else:
                        # 阶段2：进入倒车
                        control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)
                        control.reverse = True
                        # 倒车用 throttle 推进（注意限制最大倒车油门）
                        control.throttle = clamp(control.throttle + THROTTLE_UP * dt, 0.0, REVERSE_THROTTLE_MAX)

                else:
                    # 无输入：逐渐松开油门/刹车（保持自然）
                    control.throttle = max(0.0, control.throttle - THROTTLE_DOWN * dt)
                    control.brake = max(0.0, control.brake - BRAKE_DOWN * dt)
                    # reverse 保持当前状态也可以，但多数驾驶习惯是松手不强制切换
                    # 你如果希望松手自动退出倒车，可取消注释下一行：
                    # control.reverse = False

                # 应用控制
                vehicle.apply_control(control)

                # 锁相机（现在 WASD 已被吞掉，相机不会再偏移）
                follow_spectator_smooth(
                    world, vehicle,
                    height=7.5, distance=9.5, pitch=-18.0,
                    smooth=cam_smooth
                )

                world.tick()

    finally:
        # 解除吞键，避免影响系统
        try:
            for k in ["w", "a", "s", "d", "space", "esc"]:
                keyboard.unblock_key(k)
        except Exception:
            pass

        if vehicle is not None:
            vehicle.destroy()

if __name__ == "__main__":
    main()