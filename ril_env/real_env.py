import pathlib
import numpy as np
import math
import shutil
import time

from multiprocessing.managers import SharedMemoryManager
from typing import Optional
from ril_env.xarm import XArm, XArmConfig
from ril_env.multi_realsense import MultiRealsense, SingleRealsense
from ril_env.video_recorder import VideoRecorder
from ril_env.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
    align_timestamps,
)
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.replay_buffer import ReplayBuffer
from ril_env.cv2_util import get_image_transform, optimal_row_cols

# Needs to be updated
DEFAULT_OBS_KEY_MAP = {
    # robot
    "ActualTCPPose": "robot_eef_pose",
    "ActualTCPSpeed": "robot_eef_pose_vel",
    "ActualQ": "robot_joint",
    "ActualQd": "robot_joint_vel",
    # timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class Env:

    def __init__(
        self,
        output_dir,
        robot_ip,
        frequency=20,
        num_obs_steps=2,
        obs_image_resolution=(640, 480),
        max_obs_buffer_size=30,
        camera_serial_numbers=None,
        obs_key_map=DEFAULT_OBS_KEY_MAP,
        obs_float32=False,
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        tcp_offset=0.13,
        init_joints=True,
        video_capture_fps=30,
        video_capture_resolution=(1280, 720),
        record_raw_video=True,
        thread_per_video=3,
        video_crf=21,
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(1280, 720),
        shm_manager=None,
    ):

        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data["color"] = color_transform(data["color"])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution, output_res=(rw, rh), bgr_to_rgb=False
        )

        def vis_transform(data):
            data["color"] = vis_color_transform(data["color"])
            return data

        recording_transform = None
        recording_fps = video_capture_fps
        recordoring_pix_fmt = "bgr24"
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recordoring_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recordoring_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,
        )

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False)

        robot_config = XArmConfig()
        # Need interpolation controller instead
        robot = XArm(robot_config)

        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.num_obs_steps = num_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        self.lasst_realsense_data = None
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

    @property
    def is_ready(self):
        pass

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()

        self.start_time = start_time
