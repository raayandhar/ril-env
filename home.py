from ril_env.xarm_controller import XArm, XArmConfig


# Should move this into demo_real_robot.py..
def main():
    xarm_config = XArmConfig()

    with XArm(xarm_config) as arm:
        try:
            print("Homing.")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            print("Done.")


if __name__ == "__main__":
    main()
