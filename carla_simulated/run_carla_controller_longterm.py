from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from run_carla_controller import CarlaPredictiveController, VehicleSnapshot, rotation_to_quaternion, wrap_angle


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    a1 = rotation_6d[:3]
    a2 = rotation_6d[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def rotation_matrix_to_yaw(matrix: np.ndarray) -> float:
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


class CarlaLongTermPredictiveController(CarlaPredictiveController):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.long_term_seconds = args.prediction_seconds
        self.chunk_overlap = max(1, args.chunk_overlap)
        self.auto_replan = True

    def maybe_predict(self) -> None:
        if len(self.history_window) < self.runner.history_steps:
            return

        history = deque(self.history_window, maxlen=self.runner.history_steps)
        dt = self.estimate_dt(history)
        target_steps = max(int(round(self.long_term_seconds / dt)), self.runner.prediction_steps)
        full_prediction = []

        while len(full_prediction) < target_steps:
            history_arr = np.stack([snap.as_feature_vector() for snap in history])
            chunk = self.runner.predict(history_arr)
            if len(chunk) == 0:
                break

            full_prediction.extend(chunk.tolist())
            synthesized = self.rollout_chunk(list(history)[-1], chunk, dt)
            if not synthesized:
                break

            consume_count = len(synthesized)
            if len(full_prediction) < target_steps:
                consume_count = max(1, consume_count - self.chunk_overlap)

            for snapshot in synthesized[:consume_count]:
                history.append(snapshot)

        if not full_prediction:
            return

        prediction = np.asarray(full_prediction[:target_steps], dtype=np.float32)
        self.current_prediction = prediction
        self.predicted_controls = deque(
            (float(np.clip(step[0], -0.5, 1.0)), float(np.clip(step[1], -1.0, 1.0))) for step in prediction
        )
        self.autopilot_active = True
        self.draw_prediction(prediction)

    def estimate_dt(self, history: deque[VehicleSnapshot]) -> float:
        nominal_dt = 1.0 / max(self.args.fps, 1)
        if not history:
            return nominal_dt
        dts = [snap.dt for snap in history if snap.dt > 1e-3]
        if not dts:
            return nominal_dt
        estimated = float(np.median(np.asarray(dts, dtype=np.float32)))
        return float(np.clip(estimated, 0.5 * nominal_dt, 1.5 * nominal_dt))

    def rollout_chunk(
        self,
        base_snapshot: VehicleSnapshot,
        chunk_prediction: np.ndarray,
        dt: float,
    ) -> list[VehicleSnapshot]:
        snapshots: list[VehicleSnapshot] = []
        prev_dx_local = 0.0
        prev_dy_local = 0.0
        prev_speed = base_snapshot.linear_speed
        base_yaw = base_snapshot.yaw
        prev_yaw = base_yaw

        for step in chunk_prediction:
            throttle = float(step[0])
            steering = float(step[1])
            dx_local = float(step[2])
            dy_local = float(step[3])
            dz = float(step[4])
            yaw = rotation_matrix_to_yaw(rotation_6d_to_matrix(step[5:11]))
            qx, qy, qz, qw = rotation_to_quaternion(0.0, 0.0, math.degrees(yaw))
            cos_base = math.cos(base_yaw)
            sin_base = math.sin(base_yaw)
            world_x = base_snapshot.pos_x + cos_base * dx_local - sin_base * dy_local
            world_y = base_snapshot.pos_y + sin_base * dx_local + cos_base * dy_local

            delta_dx = dx_local - prev_dx_local
            delta_dy = dy_local - prev_dy_local
            delta_dist = math.hypot(delta_dx, delta_dy)
            speed = delta_dist / max(dt, 1e-3)
            accel_x = (speed - prev_speed) / max(dt, 1e-3)
            gyro_z = wrap_angle(yaw - prev_yaw) / max(dt, 1e-3)

            snapshots.append(
                VehicleSnapshot(
                    dt=dt,
                    throttle=throttle,
                    steering=steering,
                    linear_speed=speed,
                    gyro_x=0.0,
                    gyro_y=0.0,
                    gyro_z=gyro_z,
                    acc_x=accel_x,
                    acc_y=0.0,
                    # Keep gravity baseline close to real sensor distribution.
                    acc_z=base_snapshot.acc_z if abs(base_snapshot.acc_z) > 1e-3 else 9.81,
                    pos_x=world_x,
                    pos_y=world_y,
                    pos_z=base_snapshot.pos_z + dz,
                    rot_0=qx,
                    rot_1=qy,
                    rot_2=qz,
                    rot_3=qw,
                    yaw_sin=math.sin(yaw),
                    yaw_cos=math.cos(yaw),
                    yaw=yaw,
                )
            )

            prev_dx_local = dx_local
            prev_dy_local = dy_local
            prev_yaw = yaw
            prev_speed = speed

        return snapshots

    def render_overlay(self, snapshot: VehicleSnapshot) -> None:
        super().render_overlay(snapshot)
        if self.display is None:
            return
        font = pygame.font.SysFont("consolas", 20)
        target_steps = int(round(self.long_term_seconds * self.args.fps))
        text = font.render(
            f"long prediction: {self.long_term_seconds:.1f}s ({target_steps} steps target)",
            True,
            (255, 220, 120),
        )
        self.display.blit(text, (12, self.args.camera_height - 34 if self.args.enable_camera else 200))
        pygame.display.flip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn a CARLA vehicle and run long-horizon LSTM prediction for about 10 seconds."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/driving_lstm.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("artifacts/models/driving_lstm_metadata.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trigger-delay", type=float, default=0.25)
    parser.add_argument("--prefer-imu", action="store_true")
    parser.add_argument("--enable-camera", action="store_true", default=True)
    parser.add_argument("--disable-camera", action="store_false", dest="enable_camera")
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=450)
    parser.add_argument("--prediction-seconds", type=float, default=10.0)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    controller = CarlaLongTermPredictiveController(parse_args())
    controller.run()
