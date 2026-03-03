import pygame

pygame.init()
screen = pygame.display.set_mode((520, 160))
pygame.display.set_caption("pygame key test - click here then press WASD")
font = pygame.font.SysFont("consolas", 18)
clock = pygame.time.Clock()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            raise SystemExit
        if e.type in (pygame.KEYDOWN, pygame.KEYUP):
            print("EVENT:", e)

    # 关键：强制 pump 一下事件队列
    pygame.event.pump()

    focused = pygame.key.get_focused()          # 1=窗口真有键盘焦点
    active = pygame.display.get_active()        # 1=窗口处于激活状态
    keys = pygame.key.get_pressed()

    screen.fill((20, 20, 20))
    line1 = f"focused={focused}  active={active}"
    line2 = f"W={int(keys[pygame.K_w])} A={int(keys[pygame.K_a])} S={int(keys[pygame.K_s])} D={int(keys[pygame.K_d])}"
    screen.blit(font.render(line1, True, (230, 230, 230)), (12, 20))
    screen.blit(font.render(line2, True, (230, 230, 230)), (12, 55))
    screen.blit(font.render("If W/A/S/D stays 0, pygame is NOT receiving keyboard input.", True, (180, 180, 180)), (12, 95))
    pygame.display.flip()

    clock.tick(60)