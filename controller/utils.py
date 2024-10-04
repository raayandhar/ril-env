import numpy as np

def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x

def to_int16(low_byte, high_byte):
    value = (high_byte << 8) | low_byte
    if value >= 32768:
        value -= 65536
    return value

def convert(b1, b2):
    return scale_to_control(to_int16(b1, b2))

def rotation_matrix(angle, direction):
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    sina = np.sin(angle)
    cosa = np.cos(angle)
    R = np.eye(3) * cosa
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    return R
