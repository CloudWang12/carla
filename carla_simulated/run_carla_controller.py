from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import carla
import keyboard as global_keyboard
import numpy as np
import pygame
import torch

from carla_ml.config import FEATURE_COLUMNS, MODELS_DIR, TARGET_COLUMNS
from carla_ml.model import DrivingLSTM


def rotation_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    # Convert Euler angles in degrees into a quaternion tuple.
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    if qw < 0.0:
        return -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


def wrap_angle(angle: float) -> float:
    # Normalize an angle to the range [-pi, pi).
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class VehicleSnapshot:
    dt: float
    throttle: float
    steering: float
    linear_speed: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    acc_x: float
    acc_y: float
    acc_z: float
    pos_x: float
    pos_y: float
    pos_z: float
    rot_0: float
    rot_1: float
    rot_2: float
    rot_3: float
    yaw_sin: float
    yaw_cos: float
    yaw: float

    def as_feature_vector(self) -> np.ndarray:
        # Convert one vehicle snapshot into the feature order expected by the trained model.
        # 把实时车辆快照转换成与训练阶段完全一致的特征顺序。
        # 只有 FEATURE_COLUMNS 里的 15 个字段会送入模型，yaw 额外保留给调试或扩展使用。
        return np.array([getattr(self, name) for name in FEATURE_COLUMNS], dtype=np.float32)


class ModelRunner:
    def __init__(self, model_path: Path, metadata_path: Path, device: str = "cpu") -> None:
        # Load the checkpoint and metadata, then rebuild the same LSTM used during training.
        checkpoint = torch.load(model_path, map_location=device)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # 这里恢复的就是 train_lstm.py 训练并保存的 DrivingLSTM。
        # 模型超参数直接从 checkpoint 读取，保证推理结构与训练结构一致。
        self.model = DrivingLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            prediction_steps=checkpoint["prediction_steps"],
            target_size=checkpoint["target_size"],
            dropout=checkpoint["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.history_steps = metadata["history_steps"]
        self.prediction_steps = metadata["prediction_steps"]
        self.feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
        self.feature_std = np.array(metadata["feature_std"], dtype=np.float32)
        self.target_mean = np.array(metadata["target_mean"], dtype=np.float32)
        self.target_std = np.array(metadata["target_std"], dtype=np.float32)

    def predict(self, history: np.ndarray) -> np.ndarray:
        # Run one inference pass on a buffered history window and de-normalize the output.
        # history 的形状是：
        #   [history_steps, len(FEATURE_COLUMNS)]
        # 这里的数据来自 CARLA 的实时采样，不是 CSV 文件。
        # 在送入模型前，会先用训练阶段保存的 mean/std 做标准化。
        normalized = (history - self.feature_mean) / self.feature_std
        tensor = torch.tensor(normalized[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(tensor).cpu().numpy()[0]
        # 模型原始输出仍然位于标准化空间中。
        # 反标准化之后，prediction 的形状为：
        #   [prediction_steps, len(TARGET_COLUMNS)]
        # 每一行的含义是：
        #   [future_throttle, future_steering, future_dx_local, future_dy_local, future_dyaw]
        return pred * self.target_std + self.target_mean


class CarlaPredictiveController:
    def __init__(self, args: argparse.Namespace) -> None:
        # Set up CARLA connections, runtime buffers, input state, and the trained model runner.
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()
        self.vehicle: Optional[carla.Vehicle] = None
        self.imu_sensor: Optional[carla.Sensor] = None
        self.camera: Optional[carla.Sensor] = None
        self.predicted_controls: deque[tuple[float, float]] = deque()
        self.current_prediction: Optional[np.ndarray] = None
        self.history: deque[VehicleSnapshot] = deque(maxlen=256)
        self.latest_imu = {"gyro": carla.Vector3D(), "accel": carla.Vector3D()}
        self.last_manual_input_time = 0.0
        self.autopilot_active = False
        self.auto_replan = False
        self.runner = ModelRunner(args.model_path, args.metadata_path, device=args.device)
        # history_window 是推理阶段真正送进 LSTM 的滑动窗口。
        # 在它积累到 history_steps 帧之前，不会触发预测。
        self.history_window = deque(maxlen=self.runner.history_steps)
        self.display: Optional[pygame.Surface] = None
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.pressed_keys: set[int] = set()
        self.keyboard_backend = "global"
        self._global_key_latches: dict[str, bool] = {}

    def spawn_vehicle(self) -> None:
        # Spawn the ego vehicle and attach sensors needed for prediction and visualization.
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in current CARLA map.")
        vehicle_candidates = self.blueprints.filter(self.args.vehicle_filter)
        if not vehicle_candidates:
            raise RuntimeError(f"No vehicle blueprint matched filter: {self.args.vehicle_filter}")
        vehicle_bp = vehicle_candidates[0]

        start_index = self.args.spawn_index % len(spawn_points)
        selected_index: Optional[int] = None
        for offset in range(len(spawn_points)):
            candidate_index = (start_index + offset) % len(spawn_points)
            transform = spawn_points[candidate_index]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if self.vehicle is not None:
                selected_index = candidate_index
                break

        if self.vehicle is None or selected_index is None:
            raise RuntimeError("Failed to spawn vehicle on all available spawn points.")
        self.vehicle.set_autopilot(False)
        print(
            f"Spawned vehicle '{vehicle_bp.id}' at spawn point {selected_index}: "
            f"({self.vehicle.get_transform().location.x:.2f}, "
            f"{self.vehicle.get_transform().location.y:.2f}, "
            f"{self.vehicle.get_transform().location.z:.2f})"
        )

        imu_bp = self.blueprints.find("sensor.other.imu")
        imu_transform = carla.Transform(carla.Location(x=0.0, z=1.5))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        self.imu_sensor.listen(self._on_imu)

        if self.args.enable_camera:
            camera_bp = self.blueprints.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.args.camera_width))
            camera_bp.set_attribute("image_size_y", str(self.args.camera_height))
            camera_bp.set_attribute("fov", "110")
            camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-12.0))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(self._on_camera)

    def _on_imu(self, imu_data: carla.IMUMeasurement) -> None:
        # Store the latest IMU reading for the next feature snapshot.
        self.latest_imu["gyro"] = imu_data.gyroscope
        self.latest_imu["accel"] = imu_data.accelerometer

    def _on_camera(self, image: carla.Image) -> None:
        # Convert the raw CARLA image to an RGB frame that pygame can display.
        frame = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = frame.reshape((image.height, image.width, 4))[:, :, :3]
        frame = frame[:, :, ::-1]
        self.latest_camera_frame = np.swapaxes(frame, 0, 1)

    def cleanup(self) -> None:
        # Destroy all spawned actors before exit.
        actors = [self.camera, self.imu_sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def get_snapshot(self, dt: float, throttle: float, steering: float) -> VehicleSnapshot:
        # Read the current vehicle state and package it as one model input frame.
        assert self.vehicle is not None
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        accel = self.vehicle.get_acceleration()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        yaw = math.radians(transform.rotation.yaw)
        qx, qy, qz, qw = rotation_to_quaternion(
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        )
        imu_gyro = self.latest_imu["gyro"]
        imu_accel = self.latest_imu["accel"]
        acc_x = imu_accel.x if self.args.prefer_imu else accel.x
        acc_y = imu_accel.y if self.args.prefer_imu else accel.y
        acc_z = imu_accel.z if self.args.prefer_imu else accel.z
        # 每个 tick 都会把当前控制量和当前车辆状态整理成一帧样本。
        # 这些样本会被追加到 history / history_window，供后续模型推理使用。
        return VehicleSnapshot(
            dt=dt,
            throttle=throttle,
            steering=steering,
            linear_speed=speed,
            gyro_x=imu_gyro.x,
            gyro_y=imu_gyro.y,
            gyro_z=imu_gyro.z,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            pos_x=transform.location.x,
            pos_y=transform.location.y,
            pos_z=transform.location.z,
            rot_0=qx,
            rot_1=qy,
            rot_2=qz,
            rot_3=qw,
            yaw_sin=math.sin(yaw),
            yaw_cos=math.cos(yaw),
            yaw=yaw,
        )

    def manual_control(self) -> tuple[float, float, bool]:
        # Translate keyboard input into manual control and cancel autopilot when the user intervenes.
        throttle = 0.0
        brake = 0.0
        steering = 0.0
        manual = False

        if self.is_key_pressed("w", pygame.K_w):
            throttle = 0.55
            manual = True
        if self.is_key_pressed("s", pygame.K_s):
            brake = 0.35
            manual = True
        if self.is_key_pressed("a", pygame.K_a):
            steering = -0.5
            manual = True
        if self.is_key_pressed("d", pygame.K_d):
            steering = 0.5
            manual = True

        if manual:
            self.predicted_controls.clear()
            self.current_prediction = None
            self.autopilot_active = False
        return throttle - brake, steering, manual

    def is_key_pressed(self, key_name: str, pygame_key: int) -> bool:
        # Query keyboard state, preferring the global keyboard hook with a pygame fallback.
        if self.keyboard_backend == "global":
            try:
                return bool(global_keyboard.is_pressed(key_name))
            except Exception:
                self.keyboard_backend = "pygame"
        return pygame_key in self.pressed_keys

    def consume_global_keypress(self, key_name: str, pygame_key: int) -> bool:
        # Turn a held key into a one-shot keypress event.
        pressed = self.is_key_pressed(key_name, pygame_key)
        was_pressed = self._global_key_latches.get(key_name, False)
        self._global_key_latches[key_name] = pressed
        return pressed and not was_pressed

    def reset_prediction(self) -> None:
        # Clear queued predictions and return to the waiting state.
        self.predicted_controls.clear()
        self.current_prediction = None
        self.autopilot_active = False

    def maybe_predict(self) -> None:
        # Trigger the model once enough recent frames are buffered, then queue predicted controls.
        if len(self.history_window) < self.runner.history_steps:
            # 历史窗口长度不够时，模型不会执行推理。
            return
        # 当手动控制刚结束，且窗口长度足够时，
        # 会在这里把最近 history_steps 帧送进模型做一次预测。
        history_arr = np.stack([snap.as_feature_vector() for snap in self.history_window])
        prediction = self.runner.predict(history_arr)
        self.current_prediction = prediction
        # 控制器实际只消费前 2 个输出字段作为控制量：
        # step[0] -> future_throttle
        # step[1] -> future_steering
        # 后 3 个字段主要给 draw_prediction 用于轨迹可视化。
        self.predicted_controls = deque(
            (float(np.clip(step[0], -0.5, 1.0)), float(np.clip(step[1], -1.0, 1.0))) for step in prediction
        )
        self.autopilot_active = True
        self.draw_prediction(prediction)

    def draw_prediction(self, prediction: np.ndarray) -> None:
        # Draw the predicted future trajectory in the CARLA world for debugging.
        if not self.vehicle:
            return
        # 这里会把模型预测的未来轨迹画到 CARLA 世界里。
        # 使用的是 prediction 的第 3、4 列：
        #   future_dx_local / future_dy_local
        # 它们是车体局部坐标系下的位移，所以要先旋转回世界坐标系。
        transform = self.vehicle.get_transform()
        base_location = transform.location
        base_yaw = math.radians(transform.rotation.yaw)
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        previous = base_location
        for step in prediction:
            dx_local, dy_local = float(step[2]), float(step[3])
            dx_world = cos_yaw * dx_local - sin_yaw * dy_local
            dy_world = sin_yaw * dx_local + cos_yaw * dy_local
            point = carla.Location(
                x=base_location.x + dx_world,
                y=base_location.y + dy_world,
                z=base_location.z + 0.3,
            )
            self.world.debug.draw_line(previous, point, thickness=0.08, color=carla.Color(0, 255, 0), life_time=1.5)
            self.world.debug.draw_point(point, size=0.08, color=carla.Color(255, 255, 0), life_time=1.5)
            previous = point

    def run(self) -> None:
        # Main loop: handle input, run prediction when needed, apply control, and update the UI.
        pygame.init()
        window_size = (self.args.camera_width, self.args.camera_height) if self.args.enable_camera else (640, 240)
        self.display = pygame.display.set_mode(window_size)
        pygame.display.set_caption("CARLA LSTM Predictive Controller")

        self.spawn_vehicle()
        clock = pygame.time.Clock()
        prev_time = time.perf_counter()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        self.pressed_keys.add(event.key)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.reset_prediction()
                    if event.type == pygame.KEYUP and event.key in self.pressed_keys:
                        self.pressed_keys.remove(event.key)

                if self.consume_global_keypress("esc", pygame.K_ESCAPE):
                    return
                if self.consume_global_keypress("r", pygame.K_r):
                    self.reset_prediction()

                now = time.perf_counter()
                dt = max(now - prev_time, 1e-3)
                prev_time = now

                # 手动驾驶优先级最高。
                # 只要 WASD 正在按下，这个 tick 就不会触发模型预测。
                throttle, steering, manual = self.manual_control()
                if manual:
                    self.last_manual_input_time = now
                    control = carla.VehicleControl(
                        throttle=max(throttle, 0.0),
                        brake=max(-throttle, 0.0),
                        steer=steering,
                    )
                else:
                    # 模型调用时机如下：
                    # 1. 当前没有在执行上一段自动预测控制
                    # 2. 最近一次手动输入距离现在不超过 trigger_delay
                    # 3. history_window 已经积累到 history_steps 帧
                    # 满足后，maybe_predict() 会调用 LSTM，一次生成 prediction_steps 步未来预测。
                    should_trigger_after_manual = (
                        not self.autopilot_active and now - self.last_manual_input_time <= self.args.trigger_delay
                    )
                    should_auto_replan = self.auto_replan and self.autopilot_active and not self.predicted_controls
                    if should_auto_replan:
                        # Mark current segment as finished so maybe_predict can queue the next one.
                        self.autopilot_active = False
                    if should_trigger_after_manual or should_auto_replan:
                        self.maybe_predict()
                    if self.predicted_controls:
                        # 这里是真正消费模型输出的地方。
                        # 每个 tick 取出 1 步预测控制量，直到 prediction_steps 步被消耗完。
                        pred_throttle, pred_steering = self.predicted_controls.popleft()
                        control = carla.VehicleControl(
                            throttle=max(pred_throttle, 0.0),
                            brake=max(-pred_throttle, 0.0),
                            steer=pred_steering,
                        )
                    else:
                        control = carla.VehicleControl(throttle=0.0, brake=0.2, steer=0.0)

                assert self.vehicle is not None
                self.vehicle.apply_control(control)

                # 当前 tick 的控制执行完成后，才把这一帧写入历史缓存。
                # 所以下一次预测看到的是“已经真实执行过”的最新状态序列。
                snapshot = self.get_snapshot(dt, control.throttle - control.brake, control.steer)
                self.history.append(snapshot)
                self.history_window.append(snapshot)
                self.update_spectator()
                self.render_overlay(snapshot)

                print(
                    f"\rmanual={manual} autopilot={self.autopilot_active} "
                    f"speed={snapshot.linear_speed:5.2f} throttle={control.throttle:4.2f} steer={control.steer:5.2f} "
                    f"buffer={len(self.predicted_controls):02d}",
                    end="",
                    flush=True,
                )
                clock.tick(self.args.fps)
        finally:
            print()
            self.cleanup()
            pygame.quit()

    def update_spectator(self) -> None:
        # Keep the spectator camera following behind the ego vehicle.
        if not self.vehicle:
            return
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        follow_distance = 8.0
        spectator_location = carla.Location(
            x=transform.location.x - follow_distance * math.cos(yaw),
            y=transform.location.y - follow_distance * math.sin(yaw),
            z=transform.location.z + 4.0,
        )
        spectator_rotation = carla.Rotation(
            pitch=-20.0,
            yaw=transform.rotation.yaw,
            roll=0.0,
        )
        self.world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

    def render_overlay(self, snapshot: VehicleSnapshot) -> None:
        # Render the camera feed and textual runtime status in the pygame window.
        if self.display is None:
            return

        if self.args.enable_camera and self.latest_camera_frame is not None:
            surface = pygame.surfarray.make_surface(self.latest_camera_frame)
            self.display.blit(surface, (0, 0))
        else:
            self.display.fill((25, 30, 40))

        font = pygame.font.SysFont("consolas", 20)
        lines = [
            f"manual: {self.last_manual_input_time > 0 and not self.autopilot_active}",
            f"autopilot: {self.autopilot_active}",
            f"input backend: {self.keyboard_backend}",
            f"speed: {snapshot.linear_speed:.2f} m/s",
            f"throttle: {snapshot.throttle:.2f}",
            f"steer: {snapshot.steering:.2f}",
            f"pred buffer: {len(self.predicted_controls)}",
            "WASD manual, release to predict, R reset, ESC quit",
        ]
        y = 12
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            self.display.blit(text, (12, y))
            y += 26
        pygame.display.flip()


def parse_args() -> argparse.Namespace:
    # Define command-line options for CARLA connection, model files, and runtime behavior.
    parser = argparse.ArgumentParser(description="Spawn a CARLA vehicle and drive it using manual prompts + LSTM prediction.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "driving_lstm.pt")
    parser.add_argument("--metadata-path", type=Path, default=MODELS_DIR / "driving_lstm_metadata.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trigger-delay", type=float, default=0.25)
    parser.add_argument("--prefer-imu", action="store_true")
    parser.add_argument("--enable-camera", action="store_true", default=True)
    parser.add_argument("--disable-camera", action="store_false", dest="enable_camera")
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=450)
    return parser.parse_args()


if __name__ == "__main__":
    controller = CarlaPredictiveController(parse_args())
    controller.run()
