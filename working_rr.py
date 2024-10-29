import time
from threading import Thread
import cv2
import zarr
import numpy as np
import sys
import os

from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig
from ril_env.camera.rs_streamer import RealsenseStreamer

OVERLAY_ALPHA = 0.4  # higher means overlay recorded frames become more prominent

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

ext_rgb_img = None
int_rgb_img = None
ext_rgb_live = None
int_rgb_live = None


def capture_images_record():
    """
    Captures images from both external and internal cameras for recording.
    Updates global variables ext_rgb_img and int_rgb_img.
    """
    global ext_rgb_img, int_rgb_img
    while True:
        _, ext_rgb_img, _, _ = external_streamer.capture_rgbd()
        _, int_rgb_img, _, _ = internal_streamer.capture_rgbd()


def capture_images_live():
    """
    Captures live images from both external and internal cameras during replay.
    Updates global variables ext_rgb_live and int_rgb_live.
    """
    global ext_rgb_live, int_rgb_live
    while True:
        _, ext_rgb_live, _, _ = external_streamer.capture_rgbd()
        _, int_rgb_live, _, _ = internal_streamer.capture_rgbd()


if choice == "1":
    ext_rgb_img_record = []
    int_rgb_img_record = []
    dpos_record = []
    drot_record = []
    grasp_record = []
    timestamp_record = []

    print("Recording... Press Ctrl+C to stop.")

    image_thread = Thread(target=capture_images_record)
    image_thread.daemon = True
    image_thread.start()

    try:
        while True:
            loop_start_time = time.time()

            controller_state = spacemouse.get_controller_state()
            dpos = controller_state["dpos"] * xarm_cfg.position_gain
            drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
            grasp = controller_state["grasp"]

            xarm_env.step(dpos, drot, grasp)

            if ext_rgb_img is not None and int_rgb_img is not None:
                cv2.imshow("External Camera (Recording)", ext_rgb_img)
                cv2.imshow("Internal Camera (Recording)", int_rgb_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ext_rgb_img_record.append(
                ext_rgb_img.copy()
                if ext_rgb_img is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            int_rgb_img_record.append(
                int_rgb_img.copy()
                if int_rgb_img is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            dpos_record.append(dpos.copy())
            drot_record.append(drot.copy())
            grasp_record.append(grasp)
            timestamp_record.append(loop_start_time)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, control_loop_period - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        xarm_env.stop_recording()

        ext_rgb_img_record = np.array(ext_rgb_img_record, dtype=np.uint8)
        int_rgb_img_record = np.array(int_rgb_img_record, dtype=np.uint8)
        dpos_record = np.array(dpos_record, dtype=np.float32)
        drot_record = np.array(drot_record, dtype=np.float32)
        grasp_record = np.array(grasp_record, dtype=np.float32)
        timestamp_record = np.array(timestamp_record, dtype=np.float64)

        zarr_store = zarr.open("recording.zarr", mode="w")
        zarr_store.create_dataset(
            "external_images",
            data=ext_rgb_img_record,
            chunks=(1, 480, 640, 3),
            dtype="uint8",
        )
        zarr_store.create_dataset(
            "internal_images",
            data=int_rgb_img_record,
            chunks=(1, 480, 640, 3),
            dtype="uint8",
        )
        zarr_store.create_dataset(
            "dpos", data=dpos_record, chunks=(1, 3), dtype="float32"
        )
        zarr_store.create_dataset(
            "drot", data=drot_record, chunks=(1, 3), dtype="float32"
        )
        zarr_store.create_dataset(
            "grasp", data=grasp_record, chunks=(1,), dtype="float32"
        )
        zarr_store.create_dataset(
            "timestamps", data=timestamp_record, chunks=(1,), dtype="float64"
        )

        print("Recording saved to 'recording.zarr'.")

        xarm_env._arm_reset()

elif choice == "2":
    filename = input("Enter filename of the recording to load (e.g., recording.zarr): ")

    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist. Exiting.")
        sys.exit(1)

    zarr_store = zarr.open(filename, mode="r")

    ext_rgb_img_record = zarr_store["external_images"][:]
    int_rgb_img_record = zarr_store["internal_images"][:]
    dpos_record = zarr_store["dpos"][:]
    drot_record = zarr_store["drot"][:]
    grasp_record = zarr_store["grasp"][:]
    timestamp_record = zarr_store["timestamps"][:]

    print("Replaying the session... Press Ctrl+C to stop.")

    live_image_thread = Thread(target=capture_images_live)
    live_image_thread.daemon = True
    live_image_thread.start()

    should_ask = True
    try:
        for i in range(len(dpos_record)):
            if should_ask:
                continue_play = input(
                    'Enter "continue" to non-stop play the recording and enter nothing to step the frames: '
                )

            if continue_play.lower() == "continue":
                should_ask = False
            elif continue_play.lower() == "":
                print(f"Stepped frame {i}")

            loop_start_time = time.time()

            dpos = dpos_record[i]
            drot = drot_record[i]
            grasp = grasp_record[i]

            # xarm_env.step(dpos, drot, grasp)

            ext_recorded_img = ext_rgb_img_record[i]
            int_recorded_img = int_rgb_img_record[i]

            if ext_rgb_live is not None and int_rgb_live is not None:
                blended_ext_img = cv2.addWeighted(
                    ext_rgb_live, 1 - OVERLAY_ALPHA, ext_recorded_img, OVERLAY_ALPHA, 0
                )
                blended_int_img = cv2.addWeighted(
                    int_rgb_live, 1 - OVERLAY_ALPHA, int_recorded_img, OVERLAY_ALPHA, 0
                )

                cv2.imshow("External Camera (Live)", blended_ext_img)
                cv2.imshow("Internal Camera (Live)", blended_int_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if i < len(timestamp_record) - 1:
                time_diff = timestamp_record[i + 1] - timestamp_record[i]
                time.sleep(max(0.0, time_diff))
            else:
                time.sleep(control_loop_period)

        print("Replay completed.")
        xarm_env._arm_reset()
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
        xarm_env._arm_reset()
    except Exception as e:
        print(f"An error occurred during replay: {e}")
        xarm_env._arm_reset()
else:
    print("Invalid choice. Exiting.")

cv2.destroyAllWindows()
