import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time


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

        self.start_stream()

        profile = self.pipeline.get_active_profile()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = depth_sensor.get_option_value_description(
                rs.option.visual_preset, i
            )
            if visualpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)
                break

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        if self.serial_no == "317422075456":
            color_sensor.set_option(rs.option.exposure, 140)

        # Get intrinsics & extrinsics
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned_frames = self.align_to_color.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            logging.error("Failed to acquire initial frames.")
            self.stop_stream()
            raise RuntimeError("Failed to acquire initial frames.")

        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile
        )
        self.colorizer = rs.colorizer()

        self.K = np.array(
            [
                [self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                [0, 0, 1],
            ]
        )

        self.dec_filter = rs.decimation_filter()
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

    def start_stream(self, max_retries=5, delay=1):
        retries = 0
        while retries < max_retries:
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
        """
        Captures a single RGB-D frame pair.
        Returns:
            color_frame: The color frame object.
            color_image: The color image as a NumPy array.
            depth_frame: The depth frame object.
            depth_image: The depth image as a NumPy array (colorized).
        """
        retries = 0
        while retries < max_retries:
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
                depth_frame_filtered = self.filter_depth(depth_frame)
                depth_image = np.asanyarray(
                    self.colorizer.colorize(depth_frame_filtered).get_data()
                )

                return color_frame, color_image, depth_frame_filtered, depth_image
            except RuntimeError as e:
                logging.warning(f"Runtime error: {e}")
                retries += 1
                time.sleep(1)
        logging.error(
            f"Failed to capture frames from {self.serial_no} after {max_retries} attempts."
        )
        return None, None, None, None

    def filter_depth(self, depth_frame):

        # filtered = self.dec_filter.process(depth_frame)
        # filtered = self.spat_filter.process(filtered)
        # filtered = self.temp_filter.process(filtered)
        # filtered = self.hole_filling_filter.process(filtered)
        # return filtered.as_depth_frame()

        return depth_frame

    def stop_stream(self):
        self.pipeline.stop()
        logging.info(f"Camera {self.serial_no} pipeline stopped.")

    def show_image(self, image):
        cv2.imshow("img", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    realsense_streamer = RealsenseStreamer("317422075456")

    frames = []
    while True:
        _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()
        cv2.waitKey(1)
        cv2.imshow("img", rgb_image)
