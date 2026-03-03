# src/feature_runtime.py
from __future__ import annotations
import math
import carla

def rot_to_quat_wxyz(rot: carla.Rotation):
    """
    把 CARLA 的 yaw/pitch/roll 转成四元数 (w, x, y, z)
    CARLA Rotation 单位是 degree。
    """
    roll = math.radians(rot.roll)
    pitch = math.radians(rot.pitch)
    yaw = math.radians(rot.yaw)

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
    return (w, x, y, z)

def speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def build_feature_vector(
    vehicle: carla.Vehicle,
    imu_meas: carla.IMUMeasurement | None,
):
    """
    返回与你 CSV/训练一致的一帧特征，顺序必须匹配训练脚本 FEATURE_COLS：
    linear_speed,
    gyro_x,gyro_y,gyro_z,
    acc_x,acc_y,acc_z,
    pos_x,pos_y,pos_z,
    rot_0,rot_1,rot_2,rot_3
    """
    tf = vehicle.get_transform()
    loc = tf.location
    rot = tf.rotation

    spd = speed_mps(vehicle)

    if imu_meas is None:
        gx = gy = gz = 0.0
        ax = ay = az = 0.0
    else:
        # CARLA IMU: gyroscope rad/s, accelerometer m/s^2
        gx, gy, gz = float(imu_meas.gyroscope.x), float(imu_meas.gyroscope.y), float(imu_meas.gyroscope.z)
        ax, ay, az = float(imu_meas.accelerometer.x), float(imu_meas.accelerometer.y), float(imu_meas.accelerometer.z)

    qw, qx, qy, qz = rot_to_quat_wxyz(rot)

    return [
        float(spd),
        gx, gy, gz,
        ax, ay, az,
        float(loc.x), float(loc.y), float(loc.z),
        qx, qy, qz, qw
    ]