from dataclasses import dataclass

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
    :config_param ip: ip of the robot; alternative to setting in robot.conf
    :config verbose: Helpful debugging / checking print steps
    """
    tcp_maxacc: int = 5000
    position_gain: float = 10.0
    orientation_gain: float = 10.0
    alpha: float = 0.5
    control_loop_rate: int = 50
    ip: str = '192.168.1.223'
    verbose: bool = True
