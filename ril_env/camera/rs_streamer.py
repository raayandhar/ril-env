import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import imageio


class RealsenseStreamer:
    def __init__(self, serial_no):
        # in-hand : 317222072157
        # external: 317422075456

        # Configure depth and color streams
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

        # Start streaming
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        """
        self.pipe_profile = self.pipeline.start(self.config)

        profile = self.pipeline.get_active_profile()

        ## Configure depth sensor settings
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = depth_sensor.get_option_value_description(
                rs.option.visual_preset, i
            )
            if visualpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.serial_no = serial_no

        if self.serial_no == "317422075456":
            color_sensor.set_option(rs.option.exposure, 140)

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

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

    def deproject(self, px, depth_frame):
        u, v = px
        depth = depth_frame.get_distance(u, v)
        xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u, v], depth)
        return xyz

    def capture_rgb(self):
        color_frame = None
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        return color_image

    def filter_depth(self, depth_frame):
        # filtered = self.dec_filter.process(depth_frame)
        # filtered = self.spat_filter.process(filtered)

        filtered = depth_frame
        # filtered = self.hole_filling_filter.process(filtered)
        # filtered = self.temp_filter.process(filtered)
        # filtered = self.spat_filter.process(filtered)
        return filtered.as_depth_frame()

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = self.filter_depth(depth_frame)
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return color_frame, color_image, depth_frame, depth_image

    def stop_stream(self):
        self.pipeline.stop()

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
