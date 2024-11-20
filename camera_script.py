import time
import cv2
import zarr
import numpy as np
import sys
import os

from ril_env.camera import Camera, CameraConfig

# Configuration
camera_cfg = CameraConfig()
shared_directory = "/shared/recordings/"  # Update this path to your shared directory


def record_camera_session():
    print("Recording Cameras... Press Ctrl+C to stop.")

    filename = input("Enter filename to save the camera recording (save as .zarr): ")
    filename = os.path.join(shared_directory, filename)

    # Initialize cameras
    external_camera = Camera(serial_no=camera_cfg.external_serial)
    internal_camera = Camera(serial_no=camera_cfg.internal_serial)

    ext_rgb_img_record = []
    int_rgb_img_record = []
    timestamp_record = []

    try:
        while True:
            loop_start_time = time.time()

            # Check if images are available
            if (
                external_camera.color_image is not None
                and internal_camera.color_image is not None
            ):
                # Display the images
                cv2.imshow("External Camera (Recording)", external_camera.color_image)
                cv2.imshow("Internal Camera (Recording)", internal_camera.color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Record images
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

            # Maintain desired frame rate (e.g., 20 FPS)
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, 0.05 - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCamera recording stopped by user.")

    # Release camera resources
    external_camera.stop()
    internal_camera.stop()

    # Save the recording
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
    if not filename.endswith(".zarr"):
        filename += ".zarr"

    # Convert lists to numpy arrays
    ext_rgb_imgs = np.array(ext_rgb_imgs, dtype=np.uint8)
    int_rgb_imgs = np.array(int_rgb_imgs, dtype=np.uint8)
    timestamp_record = np.array(timestamp_record, dtype=np.float64)

    # Save data to Zarr format
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
    record_camera_session()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
