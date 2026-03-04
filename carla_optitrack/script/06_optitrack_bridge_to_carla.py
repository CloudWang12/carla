# script/06_optitrack_bridge_to_carla.py
import os, sys, time
from dataclasses import dataclass
from typing import Optional, Tuple

import carla
import keyboard  # pip install keyboard

# pip install natnet
from natnet import NatNetClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.carla_utils import connect, sync_mode


# ----------------------------
# 配置区：你需要根据你的 Motive 修改 IP
# ----------------------------
MOTIVE_SERVER_IP = "127.0.0.1"   # Motive所在机器IP（同机就127.0.0.1）
CLIENT_IP = "127.0.0.1"          # 你这台运行脚本的机器IP（同机就127.0.0.1）
USE_MULTICAST = True

# CARLA
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

# 运行频率（建议与你CARLA同步一致）
FIXED_DT = 0.05  # 20Hz


# ----------------------------
# 坐标映射 & 平滑
# ----------------------------
@dataclass
class AxisMapPreset:
    """
    把 Motive 的 (mx,my,mz) 映射到 CARLA 的 (x,y,z)。
    preset_name 仅用于提示
    carla_x_from/y_from/z_from: 选择 Motive 的哪个轴
    sign_x/y/z: 是否取反
    """
    preset_name: str
    carla_x_from: str  # 'x'/'y'/'z'
    carla_y_from: str
    carla_z_from: str
    sign_x: float = 1.0
    sign_y: float = 1.0
    sign_z: float = 1.0


PRESETS = [
    # 你先从这些里面试，哪个对就用哪个
    AxisMapPreset("P0: carla(x,y,z)=(mx,my,mz)", "x", "y", "z",  1,  1,  1),
    AxisMapPreset("P1: carla(x,y,z)=(mz,mx,my)", "z", "x", "y",  1,  1,  1),
    AxisMapPreset("P2: carla(x,y,z)=(mz,-mx,my)", "z", "x", "y", 1, -1, 1),
    AxisMapPreset("P3: carla(x,y,z)=(mx,-my,mz)", "x", "y", "z", 1, -1, 1),
]


def remap_pos(pos_xyz: Tuple[float, float, float], preset: AxisMapPreset) -> carla.Location:
    mx, my, mz = pos_xyz
    mp = {"x": mx, "y": my, "z": mz}
    x = preset.sign_x * mp[preset.carla_x_from]
    y = preset.sign_y * mp[preset.carla_y_from]
    z = preset.sign_z * mp[preset.carla_z_from]
    return carla.Location(x=float(x), y=float(y), z=float(z))


def lerp_loc(a: carla.Location, b: carla.Location, t: float) -> carla.Location:
    return a + (b - a) * t


@dataclass
class OptiState:
    rb_id: int
    pos_motive: Tuple[float, float, float]
    quat_motive: Tuple[float, float, float, float]  # 原样保留（不同库可能是 xyzw 或 wxyz）


class Shared:
    latest: Optional[OptiState] = None
    have_first: bool = False


def main():
    # 1) 连接 CARLA
    client = connect(CARLA_HOST, CARLA_PORT)
    world = client.get_world()

    # 2) 创建一个 marker actor（尽量用静态物体；找不到就用 spectator）
    bp_lib = world.get_blueprint_library()
    marker = None
    marker_bp = None

    # 尝试找一个简单的静态道具（不同 CARLA 版本蓝图库会不一样）
    candidates = [
        "static.prop.streetbarrier",
        "static.prop.constructioncone",
        "static.prop.trafficcone",
        "static.prop.warningconstruction",
        "static.prop.bin",
    ]
    for name in candidates:
        bp = bp_lib.find(name) if bp_lib.find(name) is not None else None
        if bp is not None:
            marker_bp = bp
            break

    if marker_bp is not None:
        # 先生成在原点附近
        marker = world.try_spawn_actor(marker_bp, carla.Transform(carla.Location(0, 0, 0.5)))
        if marker:
            print("[CARLA] spawned marker actor:", marker.type_id)
        else:
            print("[CARLA] failed to spawn marker actor, fallback to spectator.")
    else:
        print("[CARLA] no static prop blueprint found, fallback to spectator.")

    spec = world.get_spectator()

    # 3) OptiTrack / NatNet 连接
    shared = Shared()

    def on_frame(frame):
        if not frame.rigid_bodies:
            return

        rb_id, rb = next(iter(frame.rigid_bodies.items()))
        pos = rb.position
        quat = rb.rotation
        shared.latest = OptiState(rb_id=rb_id, pos_motive=(pos[0], pos[1], pos[2]), quat_motive=(quat[0], quat[1], quat[2], quat[3]))

    nat = NatNetClient(
        server_address=MOTIVE_SERVER_IP,
        client_address=CLIENT_IP,
        use_multicast=USE_MULTICAST,
    )
    nat.set_callback(on_data_frame_received=on_frame)
    nat.start()
    print("[OPTITRACK] NatNet started. Listening rigid body frames...")

    # 4) 映射/原点/平滑参数
    preset_i = 0
    preset = PRESETS[preset_i]
    print("[MAP] current preset:", preset.preset_name, preset)

    origin_carla = carla.Location(0, 0, 0)  # 你希望映射后放到CARLA哪个原点（这里用世界原点）
    origin_opti = None  # 第一次收到的 opti 位置作为原点

    smooth_alpha = 0.18  # 0~1 越小越平滑
    cur_loc = carla.Location(0, 0, 0.5)

    print("[KEYS] C recalibrate origin | M switch mapping preset | ESC quit")

    # 5) 同步模式跑起来（稳定）
    try:
        with sync_mode(world, fixed_delta_seconds=FIXED_DT):
            last_print = time.time()
            while True:
                if keyboard.is_pressed("esc"):
                    break

                if keyboard.is_pressed("m"):
                    time.sleep(0.2)  # 去抖
                    preset_i = (preset_i + 1) % len(PRESETS)
                    preset = PRESETS[preset_i]
                    print("[MAP] switched preset:", preset.preset_name, preset)

                if keyboard.is_pressed("c"):
                    time.sleep(0.2)  # 去抖
                    # 重新校准原点
                    if shared.latest is not None:
                        origin_opti = remap_pos(shared.latest.pos_motive, preset)
                        print("[CALIB] origin reset to current optitrack pos (mapped):", origin_opti)

                st = shared.latest
                if st is not None:
                    mapped = remap_pos(st.pos_motive, preset)

                    # 首帧自动校准原点
                    if origin_opti is None:
                        origin_opti = mapped
                        print("[CALIB] origin initialized:", origin_opti)

                    # 原点对齐：把 opti 的 origin 映射到 CARLA 的 origin_carla
                    target = origin_carla + (mapped - origin_opti)
                    target.z = max(0.1, target.z)  # 防止落到地下（你可按需改）

                    # 平滑
                    cur_loc = lerp_loc(cur_loc, target, smooth_alpha)

                    # 更新 CARLA marker or spectator
                    if marker is not None:
                        marker.set_transform(carla.Transform(cur_loc))
                    else:
                        # spectator 作为fallback
                        spec.set_transform(carla.Transform(cur_loc, carla.Rotation(pitch=-30, yaw=0, roll=0)))

                    # 低频打印
                    now = time.time()
                    if now - last_print > 1.0:
                        last_print = now
                        print(f"[RB {st.rb_id}] motive_pos={st.pos_motive} -> carla_pos={cur_loc}  preset={preset.preset_name}")

                world.tick()

    finally:
        try:
            nat.stop()
        except Exception:
            pass
        if marker is not None:
            marker.destroy()
        print("[DONE] stopped.")

if __name__ == "__main__":
    main()