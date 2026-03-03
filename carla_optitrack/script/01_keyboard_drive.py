# scripts/01_keyboard_drive.py
import math
import pygame
import carla

from src.carla_utils import connect, sync_mode, spawn_vehicle, follow_spectator


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    client = connect()
    world = client.get_world()

    pygame.init()
    screen = pygame.display.set_mode((420, 140))
    pygame.display.set_caption("CARLA Keyboard Drive (WASD, ESC quit)")
    font = pygame.font.SysFont("consolas", 18)

    vehicle = None
    try:
        vehicle = spawn_vehicle(world, "vehicle.tesla.model3")
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

        # “按一下给一点油门”：用 KEYDOWN 做增量
        throttle_step = 0.08
        steer_step = 0.12
        brake_step = 0.15

        with sync_mode(world, fixed_delta_seconds=0.05):
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                        if event.key == pygame.K_w:
                            control.throttle = clamp(control.throttle + throttle_step, 0.0, 1.0)
                            control.brake = 0.0
                        if event.key == pygame.K_s:
                            control.brake = clamp(control.brake + brake_step, 0.0, 1.0)
                            control.throttle = 0.0
                        if event.key == pygame.K_a:
                            control.steer = clamp(control.steer - steer_step, -1.0, 1.0)
                        if event.key == pygame.K_d:
                            control.steer = clamp(control.steer + steer_step, -1.0, 1.0)
                        if event.key == pygame.K_SPACE:
                            control = carla.VehicleControl()  # 一键清零

                vehicle.apply_control(control)
                follow_spectator(world, vehicle)

                world.tick()

                # UI
                screen.fill((20, 20, 20))
                txt = f"throttle={control.throttle:.2f} steer={control.steer:.2f} brake={control.brake:.2f}"
                screen.blit(font.render(txt, True, (230, 230, 230)), (12, 18))
                screen.blit(font.render("W/S/A/D: inc throttle/brake/steer | SPACE reset | ESC quit", True, (180, 180, 180)), (12, 55))
                pygame.display.flip()

    finally:
        if vehicle is not None:
            vehicle.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()