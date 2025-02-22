import numpy as np
import time
import scipy.spatial.transform as st
import logging

from xarm.wrapper import XArmAPI
from dataclasses import dataclass, field
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class XArmConfig:
    ip: str = "192.168.1.223"
    tcp_maxacc: int = 5000
    position_gain: float = 2.0
    orientation_gain: float = 2.0
    home_pos: List[int] = field(default_factory=lambda: [0, 0, 0, 70, 0, 70, 0])
    home_speed: float = 50.0
    verbose: bool = False


class XArm:
    def __init__(self, xarm_config: XArmConfig):
        self.config = xarm_config
        self.init = False

        self.current_position = None
        self.current_orientation = None
        self.previous_grasp = 0.0

        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

    def initialize(self):
        self.arm = XArmAPI(self.config.ip)
        arm = self.arm

        arm.connect()
        arm.clean_error()
        arm.clean_warn()

        code = arm.motion_enable(enable=True)
        if code != 0:
            logger.error(f"Error in motion_enable: {code}")
            raise RuntimeError(f"Error in motion_enable: {code}")

        arm.set_tcp_maxacc(self.config.tcp_maxacc)

        code = arm.set_mode(1)
        if code != 0:
            logger.error(f"Error in set_mode: {code}")
            raise RuntimeError(f"Error in set_mode: {code}")

        code = arm.set_state(0)
        if code != 0:
            logger.error(f"Error in set_state: {code}")
            raise RuntimeError(f"Error in set_state: {code}")

        code, state = arm.get_state()
        if code != 0:
            logger.error(f"Error getting robot state: {code}")
            raise RuntimeError(f"Error getting robot state: {code}")
        if state != 0:
            logger.error(f"Robot is not ready to move. Current state: {state}")
            raise RuntimeError(f"Robot is not ready to move. Current state: {state}")
        else:
            logger.info(f"Robot is ready to move. Current state: {state}")

        err_code, warn_code = arm.get_err_warn_code()
        if err_code != 0 or warn_code != 0:
            logger.error(
                f"Error code: {err_code}, Warning code: {warn_code}. Cleaning error and warning."
            )
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_state(0)

        code = arm.set_gripper_mode(0)
        if code != 0:
            logger.error(f"Error in set_gripper_mode: {code}")
            raise RuntimeError(f"Error in set_gripper_mode: {code}")

        code = arm.set_gripper_enable(True)
        if code != 0:
            logger.error(f"Error in set_gripper_enable: {code}")
            raise RuntimeError(f"Error in set_gripper_enable: {code}")

        code = arm.set_gripper_speed(1000)
        if code != 0:
            logger.error(f"Error in set_gripper_speed: {code}")
            raise RuntimeError(f"Error in set_gripper_speed: {code}")

        self.init = True
        time.sleep(3)
        self.home()
        time.sleep(3)
        logger.info("Successfully initialized xArm.")

    def shutdown(self):
        if not self.init:
            logger.error("shutdown() called on an uninitialized xArm.")
            return
        self.home()
        self.arm.disconnect()
        logger.info("xArm shutdown complete.")

    def home(self):
        logger.info("Homing robot.")
        if not self.init:
            logger.error("xArm not initialized.")
            raise RuntimeError("xArm not initialized.")

        arm = self.arm
        arm.set_mode(0)
        arm.set_state(0)
        code = arm.set_gripper_position(850, wait=False)
        if code != 0:
            logger.error(f"Error in set_gripper_position (open, homing): {code}")
            raise RuntimeError(f"Error in set_gripper_position (open, homing): {code}")

        code = arm.set_servo_angle(
            angle=self.config.home_pos, speed=self.config.home_speed, wait=True
        )
        if code != 0:
            logger.error(f"Error in set_servo_angle (homing): {code}")
            raise RuntimeError(f"Error in set_servo_angle (homing): {code}")
        arm.set_mode(1)
        arm.set_state(0)

        code, pose = arm.get_position()
        if code != 0:
            logger.error(f"Failed to query initial pose: {code}")
            raise RuntimeError(f"Failed to query initial pose: {code}")
        else:
            self.current_position = np.array(pose[:3])
            self.current_orientation = np.array(pose[3:])
            logger.debug(
                f"Initial pose set: pos={self.current_position}, ori={self.current_orientation}"
            )

    def step(self, dpos, drot, grasp):
        if not self.init:
            logger.error("xArm not initialized. Use it in a 'with' block")
            raise RuntimeError("xArm not initialized. Use it in a 'with' block")

        dpos *= self.config.position_gain
        drot *= self.config.orientation_gain

        curr_rot = st.Rotation.from_euler("xyz", self.current_orientation, degrees=True)
        delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
        final_rot = delta_rot * curr_rot

        self.current_orientation = final_rot.as_euler("xyz", degrees=True)
        self.current_position += dpos

        logger.debug(f"Current position: {self.current_position}")
        logger.debug(f"Current orientation: {self.current_orientation}")

        code = self.arm.set_servo_cartesian(
            np.concatenate((self.current_position, self.current_orientation)),
            is_radian=False,
        )
        if code != 0:
            logger.error(f"Error in set_servo_cartesian in step(): {code}")
            raise RuntimeError(f"Error in set_servo_cartesian in step(): {code}")

        if grasp != self.previous_grasp:
            if grasp == 1.0:
                code = self.arm.set_gripper_position(0, wait=False)
                if code != 0:
                    logger.error(f"Error in set_gripper_position (close): {code}")
                    raise RuntimeError(f"Error in set_gripper_position (close): {code}")
            else:
                code = self.arm.set_gripper_position(850, wait=False)
                if code != 0:
                    logger.error(f"Error in set_gripper_position (open): {code}")
                    raise RuntimeError(f"Error in set_gripper_position (open): {code}")
            self.previous_grasp = grasp

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
