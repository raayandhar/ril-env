import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time
from threading import Thread, Event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class Camera:
    def __init__(self, serial_no):
        self.serial_no = serial_no
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_no is not None:
            self.config.enable_device(serial_no)

        self.width = 640
        self.height = 480

        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, 30
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, 30
        )

        self.align_to_color = rs.align(rs.stream.color)

        self._stop_thread = False
        self.thread = None
        self.color_image = None
        self.depth_image = None

        self.params_ready = Event()

        self.colorizer = rs.colorizer()
        self.depth_intrin = None
        self.color_intrin = None
        self.depth_to_color_extrin = None
        self.K = None

        self.start()

        self.params_ready.wait()

    def start(self):
        self._stop_thread = False
        self.thread = Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self._stop_thread = True
        if self.thread is not None:
            self.thread.join()
        self.stop_stream()

    def _capture_frames(self):
        try:
            self.start_stream()
            self._initialize_camera_params()
            self.params_ready.set()

            while not self._stop_thread:
                color_image, depth_image = self.capture_rgbd()
                if color_image is not None and depth_image is not None:
                    self.color_image = color_image
                    self.depth_image = depth_image
                time.sleep(0.01)  # Manual thread safety?
        except Exception as e:
            logging.error(f"Error in capture thread for camera {self.serial_no}: {e}")
            self.params_ready.set()

    def start_stream(self, max_retries=5, delay=1):
        retries = 0
        while retries < max_retries and not self._stop_thread:
            try:
                self.pipeline.start(self.config)
                logging.info(f"Camera {self.serial_no} started successfully.")
                return
            except RuntimeError as e:
                logging.warning(
                    f"Attempt {retries + 1} failed to start camera {self.serial_no}: {e}"
                )
                retries += 1
                time.sleep(delay)
        logging.error(
            f"Failed to start camera {self.serial_no} after {max_retries} attempts."
        )
        raise RuntimeError(
            f"Failed to start camera {self.serial_no} after {max_retries} attempts."
        )

    def capture_rgbd(self, max_retries=3, timeout_ms=1000):
        retries = 0
        while retries < max_retries and not self._stop_thread:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if not frames:
                    raise RuntimeError("No frames received within the timeout period.")

                aligned_frames = self.align_to_color.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    raise RuntimeError("Missing depth or color frame.")

                color_image = np.asanyarray(color_frame.get_data())
                # We could use depth filtering here
                depth_image = np.asanyarray(depth_frame.get_data())

                return color_image, depth_image
            except RuntimeError as e:
                logging.warning(f"Runtime error: {e}")
                retries += 1
                time.sleep(0.1)
        logging.error(
            f"Failed to capture frames from {self.serial_no} after {max_retries} attempts."
        )
        return None, None

    def filter_depth(self, depth_frame):
        # Removed for now
        return depth_frame

    def stop_stream(self):
        try:
            self.pipeline.stop()
            logging.info(f"Camera {self.serial_no} pipeline stopped.")
        except Exception as e:
            logging.error(f"Error stopping camera {self.serial_no}: {e}")

    def _initialize_camera_params(self):
        for _ in range(10):
            if self._stop_thread:
                return
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align_to_color.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if depth_frame and color_frame:
                self.depth_intrin = (
                    depth_frame.profile.as_video_stream_profile().intrinsics
                )
                self.color_intrin = (
                    color_frame.profile.as_video_stream_profile().intrinsics
                )

                self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
                    color_frame.profile
                )

                self.K = np.array(
                    [
                        [self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                        [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                        [0, 0, 1],
                    ]
                )
                logging.info(f"Camera {self.serial_no} parameters initialized.")
                return
            time.sleep(0.1)
        logging.error(f"Failed to initialize camera parameters for {self.serial_no}.")
        raise RuntimeError(
            f"Failed to initialize camera parameters for {self.serial_no}."
        )
