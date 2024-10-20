import time
from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig
from ril_env.camera.rs_streamer import RealsenseStreamer
from threading import Thread
import cv2
import zarr
import numpy as np

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

external_streamer = RealsenseStreamer("317422075456")
internal_streamer = RealsenseStreamer("317222072157")


def capture_images():
    global ext_rgb_img, int_rgb_img
    while True:
        _, ext_rgb_img, _, _ = external_streamer.capture_rgbd()
        _, int_rgb_img, _, _ = internal_streamer.capture_rgbd()


if choice == "1":
    ext_rgb_img_record = []
    int_rgb_img_record = []

    xarm_env.start_recording()
    print("Recording... Press Ctrl+C to stop.")
    image_thread = Thread(target=capture_images)
    image_thread.daemon = True
    image_thread.start()
    try:
        while True:
            loop_start_time = time.time()

            controller_state = spacemouse.get_controller_state()
            dpos = controller_state["dpos"] * xarm_cfg.position_gain
            drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
            grasp = controller_state["grasp"]
            # xarm_env.step(dpos, drot, grasp)

            cv2.waitKey(1)
            cv2.imshow("ext", ext_rgb_img)
            cv2.imshow("int", int_rgb_img)

            ext_rgb_img_record.append(ext_rgb_img)
            int_rgb_img_record.append(int_rgb_img)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, control_loop_period - elapsed_time)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        xarm_env.stop_recording()

        """
        ext_rgb_img_record = np.array(ext_rgb_img_record)
        int_rgb_img_record = np.array(int_rgb_img_record)
        zarr.save('ext_rgb_img_record.zarr', ext_rgb_img_record)
        zarr.save('int_rgb_img_record.zarr', int_rgb_img_record)
        """

        xarm_env._arm_reset()
        filename = input(
            "Enter filename to save the recording (e.g., recording.json): "
        )
        xarm_env.save_recording(filename)

    except Exception as e:
        print(f"An error occurred: {e}")
        xarm_env.stop_recording()
        xarm_env._arm_reset()
elif choice == "2":
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
import zarr
import cv2
