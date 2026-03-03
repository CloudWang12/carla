# scripts/01_keyboard_drive_global.py
import time
from queue import SimpleQueue, Empty

import carla
from pynput import keyboard

from src.carla_utils import (
    connect,
    spawn_vehicle,
    sync_mode,
    follow_spectator_smooth,
    set_spectator_to_vehicle_once,
)

QUIT = False
ACTIONS: SimpleQueue = SimpleQueue()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def on_press(key):
    """
    这里做成“按一下加一点”的事件队列（第一版手感）
    注意：按住键时，Windows 的键盘重复也会产生 press 事件，从而持续加一点点。
    """
    global QUIT
    if key == keyboard.Key.esc:
        QUIT = True
        return False

    if key == keyboard.Key.space:
        ACTIONS.put(("reset", None))
        return

    try:
        ch = key.char.lower()
    except Exception:
        return

    if ch == "w":
        ACTIONS.put(("throttle_up", None))
    elif ch == "s":
        ACTIONS.put(("brake_up", None))
    elif ch == "a":
        ACTIONS.put(("steer_left", None))
    elif ch == "d":
        ACTIONS.put(("steer_right", None))


def on_release(key):
    # step 模式下，release 不需要处理
    pass


def main():
    global QUIT

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    client = connect()
    world = client.get_world()

    vehicle = None
    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3", spawn_index=0)

        print("[INFO] STEP control mode (like your first version).")
        print("[INFO] W/S/A/D: add a small step each press. SPACE reset. ESC quit.")
        print("[INFO] Camera is locked by script.")

        # ---- 关键：提高 tick 频率，减少“相机被 CARLA 先动一下”的可见性 ----
        with sync_mode(world, fixed_delta_seconds=1.0 / 120.0):
            # 相机先定位一次
            set_spectator_to_vehicle_once(world, vehicle)

            control = carla.VehicleControl()

            # 第一版的步长（你想要的油门速度就在这里）
            throttle_step = 0.08
            steer_step = 0.12
            brake_step = 0.15

            # 相机平滑参数（已经不抖了就保持即可）
            cam_smooth = 0.10

            while not QUIT:
                # 1) 先锁一次相机（覆盖 CARLA spectator 的输入）
                follow_spectator_smooth(world, vehicle, smooth=cam_smooth)

                # 2) 处理所有按键事件（step）
                while True:
                    try:
                        act, _ = ACTIONS.get_nowait()
                    except Empty:
                        break

                    if act == "reset":
                        control = carla.VehicleControl()

                    elif act == "throttle_up":
                        control.throttle = clamp(control.throttle + throttle_step, 0.0, 1.0)
                        control.brake = 0.0

                    elif act == "brake_up":
                        control.brake = clamp(control.brake + brake_step, 0.0, 1.0)
                        control.throttle = 0.0

                    elif act == "steer_left":
                        control.steer = clamp(control.steer - steer_step, -1.0, 1.0)

                    elif act == "steer_right":
                        control.steer = clamp(control.steer + steer_step, -1.0, 1.0)

                # 3) 应用控制
                vehicle.apply_control(control)

                # 4) tick 一步
                world.tick()

                # 5) tick 后再锁一次相机（进一步压住“那一下”）
                follow_spectator_smooth(world, vehicle, smooth=cam_smooth)

    finally:
        if vehicle is not None:
            vehicle.destroy()


if __name__ == "__main__":
    main()