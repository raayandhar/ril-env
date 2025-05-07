from ril_env.control.xarm_controller import XArm, XArmConfig


# Should move this into demo_real_robot.py...
push_home = [0, -15, 0, 10, 0, 25, 0]
def main():
    xarm_config = XArmConfig()
    xarm_config.home_pos = push_home
    xarm_config.home_on_shutdown = False
    with XArm(xarm_config) as arm:
        try:
            print("Homing.")
            arm.set_gripper_position(0)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            print("Done.")


if __name__ == "__main__":
    main()
