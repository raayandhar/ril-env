import numpy as np
import time
import sys
import os

# TODO: issue with relative import
from xarm.wrapper import XArmAPI
from .xarm_config import XArmConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


class XArmEnv:
    def __init__(self, xarm_config: XArmConfig):
        self.config = xarm_config

        self.init = False
        self.gripper_speed = self.config.gripper_speed
        self.arm = self._arm_init()
        if not self.init:
            print("Failed to initialize the arm.")
            sys.exit(1)

        self.home_pos = self.config.home_pos
        self.home_speed = self.config.home_speed

        self.previous_grasp = None
        self.gripper_open = self.config.gripper_open
        self.gripper_closed = self.config.gripper_closed
        self.control_loop_period = 1.0 / self.config.control_loop_rate
        self.dpos = None
        self.drot = None
        self.grasp = None

        self.verbose = self.config.verbose

        # TODO: recording logic should not be in env
        # temp testing for recording
        self.recording = []
        self.is_recording = False
        self.is_replaying = False
        self.replay_index = 0

        self._arm_reset()

    def step(self, dpos, drot, grasp):
        if not self.init:
            print("Error: Arm not initialized.")
            return

        if self.is_replaying:
            if self.replay_index < len(self.recording):
                action = self.recording[self.replay_index]
                dpos = np.array(action["dpos"])
                drot = np.array(action["drot"])
                grasp = action["grasp"]
                self.replay_index += 1
                time.sleep(0.005)
            else:
                self.is_replaying = False
                print("Replay finished.")
                return

        arm = self.arm
        # TODO: these are target pose, not current...
        self.current_position += dpos
        self.current_orientation += drot

        if self.verbose:
            print(f"Current position: {self.current_position} \n")
            print(f"Current orientation: {self.current_orientation} \n")

        ret = arm.set_servo_cartesian(
            np.concatenate((self.current_position, self.current_orientation)),
            is_radian=False,
        )

        if ret != 0:
            print(f"Error in set_servo_cartesian: {ret}")

        if grasp != self.previous_grasp:
            if grasp == 1.0:
                ret = arm.set_gripper_position(self.gripper_closed, wait=False)
                if ret != 0:
                    print(f"Error in set_gripper_position (close): {ret}")
            else:
                ret = arm.set_gripper_position(self.gripper_open, wait=False)
                if ret != 0:
                    print(f"Error in set_gripper_position (open): {ret}")
            self.previous_grasp = grasp

        if self.is_recording:
            self.recording.append(
                {"dpos": dpos.tolist(), "drot": drot.tolist(), "grasp": grasp}
            )

    def _arm_init(self):
        ip = self.config.ip

        arm = XArmAPI(ip)
        arm.connect()

        arm.clean_error()
        arm.clean_warn()

        ret = arm.motion_enable(enable=True)
        if ret != 0:
            print(f"Error in motion_enable: {ret}")
            sys.exit(1)

        arm.set_tcp_maxacc(self.config.tcp_maxacc)

        ret = arm.set_mode(1)  # Servo motion mode
        if ret != 0:
            print(f"Error in set_mode: {ret}")
            sys.exit(1)

        ret = arm.set_state(0)  # Ready state
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

        ret = arm.set_gripper_mode(0)
        if ret != 0:
            print(f"Error in set_gripper_mode: {ret}")
        ret = arm.set_gripper_enable(True)
        if ret != 0:
            print(f"Error in set_gripper_enable: {ret}")
        ret = arm.set_gripper_speed(self.gripper_speed)
        if ret != 0:
            print(f"Error in set_gripper_speed: {ret}")

        self.init = True
        time.sleep(1)

        return arm

    def _arm_reset(self):
        if not self.init:
            print(
                "Error initializing arm or arm was never initialized. Are you sure you're using this method correctly?"
            )
            return
        arm = self.arm
        arm.set_mode(0)
        arm.set_state(0)
        arm.set_gripper_position(self.gripper_open, wait=False)
        arm.set_servo_angle(angle=self.home_pos, speed=self.home_speed, wait=True)
        arm.set_mode(1)

        ret = arm.set_mode(1)
        if ret != 0:
            print(f"Error in set_mode: {ret}")
        ret = arm.set_state(0)
        if ret != 0:
            print(f"Error in set_state: {ret}")

        _, initial_pose = self.arm.get_position(is_radian=False)
        self.current_position = np.array(initial_pose[:3])
        self.current_orientation = np.array(initial_pose[3:])
