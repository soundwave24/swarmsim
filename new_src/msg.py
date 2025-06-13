import zmq
import time
import json
import math
import random

context = zmq.Context()
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5556")  # 수신 측에서 동일 포트 사용해야 함

def generate_position(t, pattern="circle"):
    if pattern == "circle":
        r = 5.0
        return [
            r * math.cos(t),
            r * math.sin(t),
            r * math.sin(t / 2.0)
        ]
    elif pattern == "random_walk":
        return [random.uniform(-10, 10) for _ in range(3)]
    elif pattern == "static":
        return [3.0, 4.0, 5.0]
    else:
        raise ValueError("Unknown pattern")

pattern_a = "circle"
pattern_b = "random_walk"
pattern_c = "static"

t = 0.0
dt = 0.1  # seconds = 10Hz

while True:
    ###############################################################
    # Generate positions for agents a, b, c based on their patterns
    ###############################################################
    
    # a = generate_position(t, pattern_a)
    # b = generate_position(t, pattern_b)
    # c = generate_position(t, pattern_c)

    a = [3.0, 3.0, 3.0]
    b = [-3.0, 3.0, 3.0]
    c = [4.0, -4.0, 4.0]

    message = json.dumps({
        "a": a,
        "b": b,
        "c": c
    })

    pub.send_string(message)
    print(f"[Sent] {message}")
    time.sleep(dt)
    t += dt
