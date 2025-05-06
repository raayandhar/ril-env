from ril_env.control.xarm_controller import XArm, XArmConfig


# Should move this into demo_real_robot.py..
def main():
    xarm_config = XArmConfig()

    with XArm(xarm_config):
        try:
            print("Homing.")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            print("Done.")


if __name__ == "__main__":
    main()
