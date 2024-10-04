import hid
import threading
import time
import numpy as np
import sys
import math
import os
from collections import namedtuple
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

"""
UCLA Robot Intelligence Lab

This is a simple teleop (teleoperation) script to use a SpaceMouse to control
the xArm7 robot arm. Please be careful when operating the robot arm: read the SOP first.
Additionally, please make sure your config settings are safe to use  (XArmConfig)

It may be useful to reference https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md as needed (CODE)
In case of errors, refer to https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api_code.md (ERRORS)

Written by Raayan Dhar

TODO:
- One click makes the SpaceMouse snap the gripper
- Update display controls to be accurate
"""

"""
We define some helper functions below.
These largely come from robosuite.ai (with the exception of rotation_matrix)
robosuite/robotsuite/devices/spacemouse.py
"""
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

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}

@dataclass
class SpaceMouseConfig:
    """
    Configuration class for a SpaceMouse device. It is NOT recommended to change these, with
    the exception of vendor_id and product_id, which should only be changed if using a different
    SpaceMouse device. To see the available devices, run hid.enumerate() in a Python interpreter

    :config_param pos_sensitivity: position sensitivity (do not change!)
    :config_param rot_sensitivity: rotation sensitivity (do not change!)
    :config_param verbose: adds helpful debugging print statements
    """
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    verbose: bool = False
    vendor_id: int = 9583
    product_id: int = 50741


"""
The SpaceMouse class is taken from robosuite.ai's implementation, under
robosuite/robosuite/devices/spacemouse.py.
"""

class SpaceMouse:
    def __init__(self, config: SpaceMouseConfig):
        print("Opening SpaceMouse device")
        self.pos_sensitivity = config.pos_sensitivity
        self.rot_sensitivity = config.rot_sensitivity
        self.verbose = config.verbose
        self.vendor_id = config.vendor_id
        self.product_id = config.product_id
        self.device = hid.device()

        self.device.open(self.vendor_id, self.product_id)

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, -1.0]])
        self._enabled = True

        self.lock = threading.Lock()  # Thread safety is important!

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, -1.0]])
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self._control = np.zeros(6)
        self.single_click_and_hold = False

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        with self.lock:
            control = self.control.copy()

        # Increased scaling factors
        dpos = control[:3] * 0.05 * self.pos_sensitivity
        roll, pitch, yaw = control[3:] * 0.05 * self.rot_sensitivity

        drot1 = rotation_matrix(pitch, [1.0, 0, 0])
        drot2 = rotation_matrix(roll, [0, 1.0, 0])
        drot3 = rotation_matrix(yaw, [0, 0, 1.0])

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        t_last_click = -1

        while True:
            d = self.device.read(13)
            if d is not None and self._enabled:
                if self.verbose:
                    print(f"Raw HID data: {d}")

                if self.product_id == 50741:
                    if d[0] == 1:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                    elif d[0] == 2:
                        self.roll = convert(d[1], d[2])
                        self.pitch = -convert(d[3], d[4])
                        self.yaw = -convert(d[5], d[6])

                        with self.lock:
                            self._control = [
                                self.x,
                                self.y,
                                self.z,
                                self.roll,
                                self.pitch,
                                self.yaw,
                            ]
                        if self.verbose:
                            print(f"Control values: {self._control}")
                else:
                    if d[0] == 1:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                        self.roll = convert(d[7], d[8]) 
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])

                        with self.lock:
                            self._control = [
                                self.x,
                                self.y,
                                self.z,
                                self.roll,
                                self.pitch,
                                self.yaw,
                            ]
                        if self.verbose:
                            print(f"Control values: {self._control}")

                if d[0] == 3:
                    if d[1] == 1:
                        self.single_click_and_hold = True

                    if d[1] == 0:
                        self.single_click_and_hold = False

                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self):
        return np.array(self._control)

    @property
    def control_gripper(self):
        if self.single_click_and_hold:
            return 1.0
        return 0


# At this point, we are now running a script to control the arm via SpaceMouse
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
        print("ip: ", ip)
        print(type(ip))
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('Input error, exit')
            sys.exit(1)

# Initialize the space mouse
space_mouse_cfg = SpaceMouseConfig()
space_mouse = SpaceMouse(config=space_mouse_cfg)

@dataclass
class XArmConfig:
    """
    Configuration class for some (not all!) xArm7/control parameters. The important ones are here.
    You can or should change most of these to your liking, potentially with the exception of tcp_maxacc

    :config_param tcp_maxacc: TCP (Tool Center Point, i.e., end effector) maximum acceleration
    :config_param position_gain: Increasing this value makes the position gain increase
    :config_param orientation_gain: Increasing this value makes the orientation gain increase
    :config_param alpha: This is a pseudo-smoothing factor
    :config_param control_loop_rate: Self-descriptive
    :config verbose: Helpful debugging / checking print steps
    """
    tcp_maxacc: int = 5000
    position_gain: float = 10.0
    orientation_gain: float = 10.0
    alpha: float = 0.5
    control_loop_rate: int = 50
    verbose: bool = True

xarm_cfg = XArmConfig()

arm = XArmAPI(ip)
arm.connect()

# This may be unsafe
arm.clean_error()
arm.clean_warn()

# Robot arm below:
ret = arm.motion_enable(enable=True)
if ret != 0:
    print(f"Error in motion_enable: {ret}")
    sys.exit(1)

arm.set_tcp_maxacc(xarm_cfg.tcp_maxacc)

ret = arm.set_mode(1) # This sets the mode to serve motion mode
if ret != 0:
    print(f"Error in set_mode: {ret}")
    sys.exit(1)

ret = arm.set_state(0) # This sets the state to sport (ready) state
if ret != 0:
    print(f"Error in set_state: {ret}")
    sys.exit(1)

ret, state = arm.get_state()
if ret != 0:
    print(f"Error getting robot state: {ret}")
    sys.exit(1)
if state != 0:
    print(f"Robot is not ready to move. Current state: {state}")
    sys.exit(1)
else:
    print(f"Robot is ready to move. Current state: {state}")

err_code, warn_code = arm.get_err_warn_code()
if err_code != 0 or warn_code != 0:
    print(f"Error code: {err_code}, Warning code: {warn_code}")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_state(0)

# Robot gripper below:
ret = arm.set_gripper_mode(0)  # This sets the gripper mode to location (ready)
if ret != 0:
    print(f"Error in set_gripper_mode: {ret}")
ret = arm.set_gripper_enable(True)
if ret != 0:
    print(f"Error in set_gripper_enable: {ret}")


_, initial_pose = arm.get_position(is_radian=False)
current_position = np.array(initial_pose[:3])
current_orientation = np.array(initial_pose[3:])

filtered_position = current_position.copy()
filtered_orientation = current_orientation.copy()

position_gain = xarm_cfg.position_gain
orientation_gain = xarm_cfg.orientation_gain
alpha = xarm_cfg.alpha

position_limits = {
    'x': (-500, 500),
    'y': (-500, 500),
    'z': (0, 1000)
}
orientation_limits = {
    'roll': (-180, 180),
    'pitch': (-90, 90),
    'yaw': (-180, 180)
}

control_loop_rate = xarm_cfg.control_loop_rate
control_loop_period = 1.0 / control_loop_rate

previous_grasp = None  # OTF grasp state checking

verbose = xarm_cfg.verbose

try:
    while True:
        loop_start_time = time.time()

        controller_state = space_mouse.get_controller_state()
        dpos = controller_state['dpos'] * position_gain
        drot = controller_state['raw_drotation'] * orientation_gain
        grasp = controller_state['grasp']

        if verbose:
            print(f"dpos: {dpos}")
            print(f"drot: {drot}")
            print(f"grasp: {grasp}")

        current_position += dpos
        current_orientation += drot

        # To prevent some crazy stuff from happening, we clip current position
        current_position[0] = np.clip(current_position[0], *position_limits['x'])
        current_position[1] = np.clip(current_position[1], *position_limits['y'])
        current_position[2] = np.clip(current_position[2], *position_limits['z'])

        # To prevent some crazy stuff from happening, we clip current orientation
        current_orientation[0] = np.clip(current_orientation[0], *orientation_limits['roll'])
        current_orientation[1] = np.clip(current_orientation[1], *orientation_limits['pitch'])
        current_orientation[2] = np.clip(current_orientation[2], *orientation_limits['yaw'])

        # Filtered here refers to smoothing
        filtered_position = alpha * current_position + (1 - alpha) * filtered_position
        filtered_orientation = alpha * current_orientation + (1 - alpha) * filtered_orientation

        if verbose:
            print(f"Filtered Position: {filtered_position}")
            print(f"Filtered Orientation: {filtered_orientation}")


        # !!! this is where the robot actually makes the pos/orientation change
        ret = arm.set_servo_cartesian(np.concatenate((filtered_position, filtered_orientation)), is_radian=False)

        if verbose:
            print(f"Return value from set_servo_cartesian: {ret}")

        if ret != 0:
            print(f"Error in set_servo_cartesian: {ret}")
            err_code, warn_code = arm.get_err_warn_code()
            print(f"Error code: {err_code}, Warning code: {warn_code}")

        if grasp != previous_grasp:
            if grasp == 1.0:
                ret = arm.set_gripper_position(0, wait=False) 
                if ret != 0:
                    print(f"Error in set_gripper_position (close): {ret}")
            else:
                ret = arm.set_gripper_position(850, wait=False) 
                if ret != 0:
                    print(f"Error in set_gripper_position (open): {ret}")
            previous_grasp = grasp 

        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0.0, control_loop_period - elapsed_time)
        time.sleep(sleep_time)
except KeyboardInterrupt:
    print("Teleop manually shut down!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Q: does this even do anything?
    arm.set_state(state=4)
    arm.disconnect()
