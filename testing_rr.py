import time
import cv2
import zarr
import numpy as np
import sys
import os
from threading import Thread
from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig
from ril_env.camera import Camera, CameraConfig

spacemouse_cfg = SpaceMouseConfig()
xarm_cfg = XArmConfig()
camera_cfg = CameraConfig()

OVERLAY_ALPHA = camera_cfg.overlay_alpha

xarm_env = XArmEnv(xarm_cfg)
spacemouse = SpaceMouse(spacemouse_cfg)


def record_session():
    xarm_env._arm_reset()
    print("Recording... Press Ctrl+C to stop.")

    filename = input("Enter filename to save the recording (save as .zarr): ")

    external_camera = Camera(serial_no=camera_cfg.external_serial)
    internal_camera = Camera(serial_no=camera_cfg.internal_serial)

    ext_rgb_img_record = []
    int_rgb_img_record = []
    dpos_record = []
    drot_record = []
    grasp_record = []
    timestamp_record = []

    try:
        while True:
            loop_start_time = time.time()

            controller_state = spacemouse.get_controller_state()
            dpos = controller_state["dpos"] * xarm_cfg.position_gain
            drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
            grasp = controller_state["grasp"]
            xarm_env.step(dpos, drot, grasp)

            if (
                external_camera.color_image is not None
                and internal_camera.color_image is not None
            ):
                cv2.imshow("External Camera (Recording)", external_camera.color_image)
                cv2.imshow("Internal Camera (Recording)", internal_camera.color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ext_rgb_img_record.append(
                external_camera.color_image.copy()
                if external_camera.color_image is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            int_rgb_img_record.append(
                internal_camera.color_image.copy()
                if internal_camera.color_image is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            dpos_record.append(dpos.copy())
            drot_record.append(drot.copy())
            grasp_record.append(grasp)
            timestamp_record.append(loop_start_time)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, xarm_env.control_loop_period - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    external_camera.stop()
    internal_camera.stop()

    save_recording(
        ext_rgb_img_record,
        int_rgb_img_record,
        dpos_record,
        drot_record,
        grasp_record,
        timestamp_record,
        filename,
    )
    xarm_env._arm_reset()


def replay_session():
    xarm_env._arm_reset()
    filename = input("Enter filename of the recording to load (e.g., recording.zarr): ")
    if not filename.endswith(".zarr"):
        filename += ".zarr"

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

    external_camera = Camera(serial_no=camera_cfg.external_serial)
    internal_camera = Camera(serial_no=camera_cfg.internal_serial)

    should_ask = True
    continue_play = ""

    try:
        for i in range(len(dpos_record)):
            ext_recorded_img = ext_rgb_img_record[i]
            int_recorded_img = int_rgb_img_record[i]

            if (
                external_camera.color_image is not None
                and internal_camera.color_image is not None
            ):
                blended_ext_img = cv2.addWeighted(
                    external_camera.color_image,
                    1 - OVERLAY_ALPHA,
                    ext_recorded_img,
                    OVERLAY_ALPHA,
                    0,
                )
                blended_int_img = cv2.addWeighted(
                    internal_camera.color_image,
                    1 - OVERLAY_ALPHA,
                    int_recorded_img,
                    OVERLAY_ALPHA,
                    0,
                )

                cv2.imshow("External Camera (Live)", blended_ext_img)
                cv2.imshow("Internal Camera (Live)", blended_int_img)

            dpos = dpos_record[i]
            drot = drot_record[i]
            grasp = grasp_record[i]

            xarm_env.step(dpos, drot, grasp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if i < len(timestamp_record) - 1:
                time_diff = timestamp_record[i + 1] - timestamp_record[i]
                time.sleep(max(0.0, time_diff))
            else:
                time.sleep(xarm_env.control_loop_period)
        print("Replay completed.")
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    except Exception as e:
        print(f"An error occurred during replay: {e}")
    finally:
        external_camera.stop()
        internal_camera.stop()
        xarm_env._arm_reset()


def select_frames_overlay():
    xarm_env._arm_reset()
    filename = input("Enter filename of the recording to load (e.g., recording.zarr): ")
    if not filename.endswith(".zarr"):
        filename += ".zarr"

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

    external_camera = Camera(serial_no=camera_cfg.external_serial)
    internal_camera = Camera(serial_no=camera_cfg.internal_serial)

    should_ask = True
    continue_play = ""

    try:
        ext_recorded_img = None
        int_recorded_img = None

        try:
            for i in range(len(dpos_record)):
                ext_recorded_img = ext_rgb_img_record[i]
                int_recorded_img = int_rgb_img_record[i]

                cv2.imshow("External Camera (Overlay)", ext_recorded_img)
                cv2.imshow("Internal Camera (Overlay)", int_recorded_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if i < len(timestamp_record) - 1:
                    time_diff = timestamp_record[i + 1] - timestamp_record[i]
                    time.sleep(max(0.0, time_diff))
                else:
                    time.sleep(xarm_env.control_loop_period)
        except KeyboardInterrupt:
            while True:
                if (
                    external_camera.color_image is not None
                    and internal_camera.color_image is not None
                ):
                    blended_ext_img = cv2.addWeighted(
                        external_camera.color_image,
                        1 - OVERLAY_ALPHA,
                        ext_recorded_img,
                        OVERLAY_ALPHA,
                        0,
                    )
                    blended_int_img = cv2.addWeighted(
                        internal_camera.color_image,
                        1 - OVERLAY_ALPHA,
                        int_recorded_img,
                        OVERLAY_ALPHA,
                        0,
                    )

                    cv2.imshow("External Camera (Overlay)", blended_ext_img)
                    cv2.imshow("Internal Camera (Overlay)", blended_int_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if i < len(timestamp_record) - 1:
                    time_diff = timestamp_record[i + 1] - timestamp_record[i]
                    time.sleep(max(0.0, time_diff))
                else:
                    time.sleep(xarm_env.control_loop_period)
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    except Exception as e:
        print(f"An error occurred during replay: {e}")
    finally:
        external_camera.stop()
        internal_camera.stop()
        xarm_env._arm_reset()


def save_recording(
    ext_rgb_imgs,
    int_rgb_imgs,
    dpos_record,
    drot_record,
    grasp_record,
    timestamp_record,
    filename,
):
    if not filename.endswith(".zarr"):
        filename += ".zarr"

    ext_rgb_imgs = np.array(ext_rgb_imgs, dtype=np.uint8)
    int_rgb_imgs = np.array(int_rgb_imgs, dtype=np.uint8)
    dpos_record = np.array(dpos_record, dtype=np.float32)
    drot_record = np.array(drot_record, dtype=np.float32)
    grasp_record = np.array(grasp_record, dtype=np.float32)
    timestamp_record = np.array(timestamp_record, dtype=np.float64)

    zarr_store = zarr.open(filename, mode="w")
    zarr_store.create_dataset(
        "external_images", data=ext_rgb_imgs, chunks=(1, 480, 640, 3), dtype="uint8"
    )
    zarr_store.create_dataset(
        "internal_images", data=int_rgb_imgs, chunks=(1, 480, 640, 3), dtype="uint8"
    )
    zarr_store.create_dataset("dpos", data=dpos_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("drot", data=drot_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("grasp", data=grasp_record, chunks=(1,), dtype="float32")
    zarr_store.create_dataset(
        "timestamps", data=timestamp_record, chunks=(1,), dtype="float64"
    )

    print(f"Recording saved to '{filename}'.")


def main():
    print("Select an option:")
    print("1. Record a new session")
    print("2. Replay a session")
    print("3. Select frames and overlay")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == "1":
        record_session()
    elif choice == "2":
        replay_session()
    elif choice == "3":
        select_frames_overlay()
    else:
        print("Invalid choice. Exiting.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
