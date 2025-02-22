import time

from ril_env.spacemouse import Spacemouse
from ril_env.xarm import XArm, XArmConfig


def main():

    xarm_config = XArmConfig()

    with Spacemouse(deadzone=0.4) as sm, XArm(xarm_config) as arm:
        try:
            while True:
                loop_start = time.monotonic()

                time.sleep(0.01)

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                if sm.is_button_pressed(1):
                    arm.home()
                    continue

                arm.step(dpos, drot, grasp)

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, 0.02 - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
