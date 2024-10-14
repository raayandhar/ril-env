import time
from robot_env.xarm_env import XArmEnv
from robot_env.xarm_config import XArmConfig
from controller.spacemouse_config import SpaceMouseConfig
from controller.spacemouse_controller import SpaceMouse

# should use click for CLI args instead...
spacemouse_cfg = SpaceMouseConfig()
xarm_cfg = XArmConfig()

xarm_env = XArmEnv(xarm_cfg)
spacemouse = SpaceMouse(spacemouse_cfg)

control_loop_rate = xarm_cfg.control_loop_rate
control_loop_period = 1.0 / control_loop_rate
xarm_env._arm_reset()

print("Select an option:")
print("1. Record a new session")
print("2. Replay a session")
choice = input("Enter your choice (1 or 2): ")

if choice == '1':
    xarm_env.start_recording()
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            loop_start_time = time.time()

            controller_state = spacemouse.get_controller_state()
            dpos = controller_state['dpos'] * xarm_cfg.position_gain
            drot = controller_state['raw_drotation'] * xarm_cfg.orientation_gain
            grasp = controller_state['grasp']
            xarm_env.step(dpos, drot, grasp)
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, control_loop_period - elapsed_time)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        xarm_env.stop_recording()
        xarm_env._arm_reset()
        filename = input("Enter filename to save the recording (e.g., recording.json): ")
        xarm_env.save_recording(filename)
    except Exception as e:
        print(f"An error occurred: {e}")
        xarm_env.stop_recording()
        xarm_env._arm_reset()
elif choice == '2':
    filename = input("Enter filename of the recording to load (e.g., recording.json): ")
    xarm_env.load_recording(filename)
    xarm_env.start_replay()
    print("Replaying the session... Press Ctrl+C to stop.")
    try:
        while xarm_env.is_replaying:
            xarm_env.step(None, None, None)
            time.sleep(control_loop_period)
        xarm_env._arm_reset()
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
        xarm_env._arm_reset()
    except Exception as e:
        print(f"An error occurred during replay: {e}")
        xarm_env._arm_reset()
else:
    print("Invalid choice. Exiting.")
