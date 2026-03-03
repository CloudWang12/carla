# src/carla_utils.py
from __future__ import annotations

import contextlib
import carla


def connect(host: str = "127.0.0.1", port: int = 2000, timeout: float = 10.0) -> carla.Client:
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def spawn_vehicle(
    world: carla.World,
    blueprint_filter: str = "vehicle.tesla.model3",
    spawn_index: int = 0,
) -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter(blueprint_filter)
    if not bps:
        raise RuntimeError(f"No blueprint matched: {blueprint_filter}")
    bp = bps[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map has no spawn points.")

    spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
    vehicle = world.try_spawn_actor(bp, spawn_points[spawn_index])
    if vehicle is None:
        # 如果该点被占用，尝试其他点
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(bp, sp)
            if vehicle is not None:
                break
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle (all spawn points occupied?).")
    return vehicle


@contextlib.contextmanager
def sync_mode(world: carla.World, fixed_delta_seconds: float = 1.0 / 60.0, no_rendering: bool = False):
    """
    同步模式：适合做稳定 demo/训练/复现。
    注意：如果你之后要接传感器/录制数据，同步模式更好控。
    """
    original_settings = world.get_settings()
    settings = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=fixed_delta_seconds,
        no_rendering_mode=no_rendering,
    )
    world.apply_settings(settings)
    try:
        yield
    finally:
        world.apply_settings(original_settings)


def _yaw_lerp_deg(cur_yaw: float, tgt_yaw: float, t: float) -> float:
    """
    处理 yaw 在 [-180,180] 的环绕插值，避免跨 180 度时抖一下。
    """
    diff = (tgt_yaw - cur_yaw + 180.0) % 360.0 - 180.0
    return cur_yaw + diff * t


def follow_spectator_smooth(
    world: carla.World,
    vehicle: carla.Vehicle,
    height: float = 7.5,
    distance: float = 9.5,
    pitch: float = -18.0,
    smooth: float = 0.10,
):
    """
    平滑跟车相机：每 tick 调一次。
    smooth: 0~1，越小越平滑但跟随更慢。建议 0.08~0.18。
    """
    spec = world.get_spectator()

    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()

    desired_loc = vt.location - forward * distance
    desired_loc.z += height
    desired_rot = carla.Rotation(pitch=pitch, yaw=vt.rotation.yaw, roll=0.0)

    cur_tf = spec.get_transform()

    # 位置插值
    new_loc = cur_tf.location + (desired_loc - cur_tf.location) * smooth

    # 角度插值（yaw 用环绕插值）
    new_yaw = _yaw_lerp_deg(cur_tf.rotation.yaw, desired_rot.yaw, smooth)
    new_pitch = cur_tf.rotation.pitch + (desired_rot.pitch - cur_tf.rotation.pitch) * smooth

    spec.set_transform(carla.Transform(new_loc, carla.Rotation(pitch=new_pitch, yaw=new_yaw, roll=0.0)))


def set_spectator_to_vehicle_once(
    world: carla.World,
    vehicle: carla.Vehicle,
    height: float = 7.5,
    distance: float = 9.5,
    pitch: float = -18.0,
):
    """
    只定位一次相机到车后方（不平滑）。
    """
    spec = world.get_spectator()
    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()
    loc = vt.location - forward * distance
    loc.z += height
    spec.set_transform(carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=vt.rotation.yaw, roll=0.0)))