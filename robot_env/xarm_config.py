from dataclasses import dataclass, field
from typing import List

@dataclass
class XArmConfig:
    """
    Configuration class for some (not all!) xArm7/control parameters. The important ones are here.
    You can or should change most of these to your liking, potentially with the exception of tcp_maxacc.

    :config_param tcp_maxacc: TCP (Tool Center Point, i.e., end effector) maximum acceleration
    :config_param position_gain: Increasing this value makes the position gain increase
    :config_param orientation_gain: Increasing this value makes the orientation gain increase
    :config_param alpha: This is a pseudo-smoothing factor
    :config_param control_loop_rate: Self-descriptive
    :config_param ip: IP of the robot; alternative to setting in robot.conf
    :config_param home_pos: The home/default position of the robot arm
    :config_param home_speed: How fast the robot arm should move when reseting
    :config_param gripper_speed: How fast the gripper should be opening and closing
    :config_param gripper_open: position of the open gripper (do not change unless really necessary!)
    :config_param gripper_closed: position of the closed gripper (do not change unless really necessary!)
    :config_param verbose: Helpful debugging / checking print steps
    """
    tcp_maxacc: int = 5000
    position_gain: float = 10.0
    orientation_gain: float = 10.0
    alpha: float = 0.5
    control_loop_rate: int = 50
    ip: str = '192.168.1.223'
    home_pos: List[int] = field(default_factory=lambda: [0, 0, 0, 70, 0, 70, 0])
    home_speed: float = 50.0
    gripper_speed: int = 1000
    gripper_open: int = 850
    gripper_closed: int = 0
    verbose: bool = True
