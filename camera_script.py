import time
import cv2
import zarr
import numpy as np
import os
import argparse
import signal

from ril_env.camera import Camera, CameraConfig

camera_cfg = CameraConfig()
OVERLAY_ALPHA = camera_cfg.overlay_alpha


def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Stopping camera recording.")
    global stop_recording
    stop_recording = True


def record_camera_session(filename):
    global stop_recording
    stop_recording = False

    print("Recording Cameras... Press Ctrl+C to stop.")

    external_camera = Camera(serial_no=camera_cfg.external_serial)
    internal_camera = Camera(serial_no=camera_cfg.internal_serial)

    ext_rgb_img_record = []
    int_rgb_img_record = []
    timestamp_record = []

    try:
        while not stop_recording:
            loop_start_time = time.time()

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
            timestamp_record.append(loop_start_time)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, 0.05 - elapsed_time)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"\nAn error occurred during camera recording: {e}")

    finally:
        external_camera.stop()
        internal_camera.stop()
        cv2.destroyAllWindows()

        save_camera_recording(
            ext_rgb_img_record,
            int_rgb_img_record,
            timestamp_record,
            filename,
        )


def save_camera_recording(
    ext_rgb_imgs,
    int_rgb_imgs,
    timestamp_record,
    filename,
):
    recordings_dir = "/home/u-ril/Github/ril-env/recordings"  # This is manually set
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    filename = os.path.join(recordings_dir, filename)

    if not filename.endswith(".zarr"):
        filename += ".zarr"

    print(f"Camera recording will be saved to '{filename}'.")

    ext_rgb_imgs = np.array(ext_rgb_imgs, dtype=np.uint8)
    int_rgb_imgs = np.array(int_rgb_imgs, dtype=np.uint8)
    timestamp_record = np.array(timestamp_record, dtype=np.float64)

    zarr_store = zarr.open(filename, mode="w")
    zarr_store.create_dataset(
        "external_images", data=ext_rgb_imgs, chunks=(1, 480, 640, 3), dtype="uint8"
    )
    zarr_store.create_dataset(
        "internal_images", data=int_rgb_imgs, chunks=(1, 480, 640, 3), dtype="uint8"
    )
    zarr_store.create_dataset(
        "timestamps", data=timestamp_record, chunks=(1,), dtype="float64"
    )

    print(f"Camera recording saved to '{filename}'.")


def main():
    parser = argparse.ArgumentParser(description="Camera Recording Script")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Filename to save the camera recording",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    record_camera_session(args.filename)


if __name__ == "__main__":
    main()
