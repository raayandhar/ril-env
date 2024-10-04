import hid
import threading
import time
import numpy as np
import sys
import math
import os
from collections import namedtuple
from .utils import scale_to_control, to_int16, convert, rotation_matrix
from .spacemouse_config import SpaceMouseConfig

from xarm.wrapper import XArmAPI

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

