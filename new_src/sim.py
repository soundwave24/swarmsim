import zmq
import json
import math
import pygame
import threading
import time

# -- 설정 --
AGENTS = ['a', 'b', 'c']
speed = 1.0  # m/s
target_update_interval = 0.1  # 10Hz

# -- 초기 위치/타겟 정의 --
positions = {k: [0.0, 0.0, 0.0] for k in AGENTS}
targets = {k: [0.0, 0.0, 0.0] for k in AGENTS}

# -- ZeroMQ Subscriber 설정 --
context = zmq.Context()
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:5556")  # Publisher 주소
sub.setsockopt_string(zmq.SUBSCRIBE, "")

# -- 메시지 수신 스레드 --
def listen_for_targets():
    while True:
        try:
            message = sub.recv_string()
            data = json.loads(message)
            for k in AGENTS:
                if k in data:
                    targets[k] = data[k]
        except Exception as e:
            print(f"[Error receiving]: {e}")

# -- 비동기 수신 시작 --
threading.Thread(target=listen_for_targets, daemon=True).start()

# -- 시각화 초기화 (2D: X vs Y) --
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Triangle Agents")
clock = pygame.time.Clock()

def world_to_screen(pos, scale=20, offset=(400, 400)):
    x, y = pos[0], pos[1]
    return int(x * scale + offset[0]), int(-y * scale + offset[1])

# -- 메인 루프 --
running = True
while running:
    dt = clock.tick(60) / 1000  # Delta time in seconds
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 에이전트 이동
    for k in AGENTS:
        px, py, pz = positions[k]
        tx, ty, tz = targets[k]
        dx, dy, dz = tx - px, ty - py, tz - pz
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if dist > 1e-3:
            vx, vy, vz = dx / dist * speed, dy / dist * speed, dz / dist * speed
            positions[k][0] += vx * dt
            positions[k][1] += vy * dt
            positions[k][2] += vz * dt

    # 화면 그리기
    screen.fill((30, 30, 30))
    colors = {"a": (255, 0, 0), "b": (0, 255, 0), "c": (0, 150, 255)}

    for k in AGENTS:
        sx, sy = world_to_screen(positions[k])
        pygame.draw.circle(screen, colors[k], (sx, sy), 10)
        # 타겟 위치 표시
        txs, tys = world_to_screen(targets[k])
        pygame.draw.circle(screen, (colors[k][0]//2, colors[k][1]//2, colors[k][2]//2), (txs, tys), 5)

    pygame.display.flip()

pygame.quit()
