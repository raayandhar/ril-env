import pathlib
import numpy as np
import shutil
import time
import logging

from multiprocessing.managers import SharedMemoryManager

from ril_env.xarm import XArm, XArmConfig
from ril_env.multi_realsense import MultiRealsense, SingleRealsense
from ril_env.video_recorder import VideoRecorder
from ril_env.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
)
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.replay_buffer import ReplayBuffer
from ril_env.cv2_util import get_image_transform, optimal_row_cols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OBS_KEY_MAP = {
    # Example placeholders:
    "xarm_eef_pose": "robot_eef_pose",
    "xarm_gripper": "robot_gripper",
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}

class RealEnv:
    """
    Simplified RealEnv that does not block/wait on realsense camera readiness.
    It just calls realsense.start(wait=False) and proceeds immediately.
    If the cameras take time to become ready, your get_obs() might initially return older frames or fewer frames.
    """

    def __init__(
        self,
        output_dir,
        robot_ip,
        frequency=10,
        n_obs_steps=2,
        obs_image_resolution=(640, 480),
        max_obs_buffer_size=30,
        camera_serial_numbers=None,
        obs_key_map=DEFAULT_OBS_KEY_MAP,
        obs_float32=False,
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        tcp_offset=0.13,
        init_joints=False,
        video_capture_fps=30,
        video_capture_resolution=(1280, 720),
        record_raw_video=True,
        thread_per_video=2,
        video_crf=21,
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(1280, 720),
        shm_manager=None,
    ):
        logger.info("Initializing RealEnv (no camera waits).")

        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        logger.info(f"RealEnv output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        logger.info(f"Replay buffer path: {zarr_path}")
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            logger.info("Started local SharedMemoryManager.")
        self.shm_manager = shm_manager

        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()
        logger.info(f"Camera serials: {camera_serial_numbers}")

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        color_transform = color_tf
        if obs_float32:
            def float_transform(img):
                return color_tf(img).astype(np.float32) / 255.0
            color_transform = float_transform

        def transform(data):
            if "color" in data:
                data["color"] = color_transform(data["color"])
            return data

        # multi-cam visual transform
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False,
        )
        def vis_transform(data):
            if "color" in data:
                data["color"] = vis_color_transform(data["color"])
            return data

        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = "bgr24"
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recording_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
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
                window_name="Multi Cam Vis",
                rgb_to_bgr=False,
            )

        robot_config = XArmConfig(ip=robot_ip)
        self.robot = XArm(robot_config)

        # Store
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.realsense = realsense
        self.multi_cam_vis = multi_cam_vis

        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        self.obs_float32 = obs_float32
        self.tcp_offset = tcp_offset
        self.init_joints = init_joints

        # accumulators for episodes
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        self.last_realsense_data = None
        self.start_time = None

        logger.info("RealEnv init complete (no camera waits).")

    @property
    def is_ready(self):
        """
        We won't block on cameras. If you want to see if cameras are streaming,
        you can check realsense.is_ready, but we won't enforce it.
        The robot is considered init if self.robot.init is True.
        """
        # We'll just return True if the robot is init. 
        # Or we can also do `and self.realsense.is_ready` if you want to consider camera readiness.
        return self.robot.init

    def start(self, wait=True):
        logger.info("RealEnv.start() called (no wait on cameras).")
        # Start cameras non-blocking
        self.realsense.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        self.robot.initialize()
        if self.init_joints:
            self.robot.home()

        logger.info(f"RealEnv start done. is_ready={self.is_ready}")

    def stop(self, wait=True):
        logger.info("RealEnv.stop() called (no wait).")
        self.end_episode()  
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.realsense.stop(wait=False)
        self.robot.shutdown()
        logger.info("RealEnv stop done.")

    def __enter__(self):
        logger.info("Entering RealEnv context manager (no wait).")
        self.start(wait=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting RealEnv context manager (no wait).")
        self.stop(wait=False)

    def get_obs(self) -> dict:
        """
        We'll just try to pull the last n_obs_steps frames from realsense, 
        plus the current xArm state. 
        No blocking if cameras haven't started streaming yet.
        """
        # We don't check self.is_ready, because we don't want to block.
        k = max(self.n_obs_steps, 1)
        logger.debug(f"RealEnv.get_obs() retrieving {k} frames from realsense.")
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # Robot
        code, pose = self.robot.arm.get_position()
        xarm_pose = np.array(pose[:6], dtype=np.float32)
        xarm_gripper = np.array([self.robot.previous_grasp], dtype=np.float32)

        stacked_pose = np.tile(xarm_pose[None, :], (self.n_obs_steps, 1))
        stacked_gripper = np.tile(xarm_gripper[None, :], (self.n_obs_steps, 1))

        camera_obs = {}
        i = 0
        for val in self.last_realsense_data.values():
            camera_obs[f"camera_{i}"] = val["color"][-self.n_obs_steps :]
            i += 1

        obs_data = dict(camera_obs)
        obs_data["robot_eef_pose"] = stacked_pose
        obs_data["robot_gripper"] = stacked_gripper
        now = time.time()
        obs_data["timestamp"] = np.linspace(
            now - (self.n_obs_steps - 1) / self.frequency, now, self.n_obs_steps
        )

        # If episode accumulators exist, store single-step
        if self.obs_accumulator is not None:
            now_array = np.array([now], dtype=np.float64)
            raw_obs = {
                "xarm_eef_pose": xarm_pose[None, :],
                "xarm_gripper": xarm_gripper[None, :],
            }
            self.obs_accumulator.put(raw_obs, now_array)

        return obs_data

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray):
        """
        Just apply the first action immediately. 
        No blocking or advanced scheduling.
        """
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        if len(actions) > 0:
            act = actions[0]
            dpos = act[:3]
            drot = act[3:6] if len(act) >= 6 else np.zeros(3)
            grasp = act[6] if len(act) >= 7 else 0.0
            self.robot.step(dpos, drot, grasp)

        if self.action_accumulator is not None:
            self.action_accumulator.put(actions, timestamps)

    def get_robot_state(self):
        code, pose = self.robot.arm.get_position()
        xarm_pose = np.array(pose[:6], dtype=np.float32)
        gripper = self.robot.previous_grasp
        return {"eef_pose": xarm_pose, "gripper": gripper}

    def start_episode(self, start_time=None):
        """
        Start a new demonstration episode: 
        re-initialize accumulators, start camera recording, etc.
        """
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        # Add a new episode subfolder
        episode_id = self.replay_buffer.n_episodes
        ep_dir = self.video_dir.joinpath(str(episode_id))
        ep_dir.mkdir(parents=True, exist_ok=True)

        n_cams = self.realsense.n_cameras
        paths = []
        for i in range(n_cams):
            paths.append(str(ep_dir.joinpath(f"{i}.mp4").absolute()))

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(paths, start_time=start_time)

        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        logger.info(f"Episode {episode_id} started with no camera waits at t={start_time:.3f}")

    def end_episode(self):
        self.realsense.stop_recording()
        if self.obs_accumulator is None:
            return

        obs_data = self.obs_accumulator.data
        obs_timestamps = self.obs_accumulator.timestamps
        actions = self.action_accumulator.actions
        action_timestamps = self.action_accumulator.timestamps
        stages = self.stage_accumulator.actions

        n_steps = min(len(obs_timestamps), len(action_timestamps))
        if n_steps > 0:
            ep = {}
            ep["timestamp"] = obs_timestamps[:n_steps]
            ep["action"] = actions[:n_steps]
            ep["stage"] = stages[:n_steps]
            for k, arr in obs_data.items():
                ep[k] = arr[:n_steps]
            self.replay_buffer.add_episode(ep, compressors="disk")
            ep_id = self.replay_buffer.n_episodes - 1
            logger.info(f"Episode {ep_id} saved with {n_steps} steps (no wait).")

        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

    def drop_episode(self):
        """
        Stop any current episode, then drop from the replay buffer 
        and remove the last video directory.
        """
        self.end_episode()
        self.replay_buffer.drop_episode()
        ep_id = self.replay_buffer.n_episodes
        ep_dir = self.video_dir.joinpath(str(ep_id))
        if ep_dir.exists():
            shutil.rmtree(str(ep_dir))
        logger.info(f"Episode {ep_id} dropped (no wait).")

"""
import pathlib
import numpy as np
import math
import shutil
import time
import logging

from multiprocessing.managers import SharedMemoryManager
from typing import Optional

# xArm code
from ril_env.xarm import XArm, XArmConfig

# Realsense code
from ril_env.multi_realsense import MultiRealsense, SingleRealsense
from ril_env.video_recorder import VideoRecorder

# Time accumulators & replay
from ril_env.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
)
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.replay_buffer import ReplayBuffer

# Utilities for transforms
from ril_env.cv2_util import get_image_transform, optimal_row_cols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OBS_KEY_MAP = {
    "xarm_eef_pose": "robot_eef_pose",
    "xarm_gripper": "robot_gripper",
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}

class RealEnv:


    def __init__(
        self,
        output_dir,
        robot_ip,
        frequency=10,   # "env step" frequency
        n_obs_steps=2,
        obs_image_resolution=(640, 480),
        max_obs_buffer_size=30,
        camera_serial_numbers=None,
        obs_key_map=DEFAULT_OBS_KEY_MAP,
        obs_float32=False,
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        tcp_offset=0.13,
        init_joints=False,
        video_capture_fps=30,
        video_capture_resolution=(1280, 720),
        record_raw_video=True,
        thread_per_video=2,
        video_crf=21,
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(1280, 720),
        shm_manager=None,
    ):
        logger.info("Initializing RealEnv...")

        assert frequency <= video_capture_fps, f"frequency={frequency} cannot exceed video_capture_fps={video_capture_fps}"
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps

        # Output path
        output_dir = pathlib.Path(output_dir)
        logger.info(f"RealEnv output directory: {output_dir}")
        assert output_dir.parent.is_dir(), f"Parent of {output_dir} not found!"
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        # Replay buffer
        zarr_path = output_dir.joinpath("replay_buffer.zarr").absolute()
        logger.info(f"Replay buffer path: {zarr_path}")
        replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="a")

        # SharedMemory
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            logger.info("Started local SharedMemoryManager.")
        self.shm_manager = shm_manager

        # If camera_serial_numbers is None, we auto-detect
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()
        logger.info(f"Camera serials: {camera_serial_numbers}")

        # Build transforms
        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        color_transform = color_tf
        if obs_float32:
            def float_transform(img):
                return color_tf(img).astype(np.float32) / 255.0
            color_transform = float_transform

        def transform(data):
            if "color" in data:
                data["color"] = color_transform(data["color"])
            return data

        # Visualization transform
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False,
        )
        def vis_transform(data):
            if "color" in data:
                data["color"] = vis_color_transform(data["color"])
            return data

        # Decide how to record raw frames or downsample:
        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = "bgr24"
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recording_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,
        )

        # Create MultiRealsense
        self.realsense = MultiRealsense(
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
            verbose=True,   # enable debug prints from SingleRealsense
        )

        self.multi_cam_vis = None
        if enable_multi_cam_vis:
            self.multi_cam_vis = MultiCameraVisualizer(
                realsense=self.realsense,
                row=row,
                col=col,
                window_name="Multi Cam Vis",
                rgb_to_bgr=False,
            )

        # Create xArm
        xarm_config = XArmConfig(ip=robot_ip)
        self.robot = XArm(xarm_config)

        # Store instance fields
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.obs_key_map = obs_key_map
        self.obs_float32 = obs_float32
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.tcp_offset = tcp_offset
        self.init_joints = init_joints

        self.last_realsense_data = None
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        self.start_time = None

        logger.info("RealEnv init complete.")

    @property
    def is_ready(self):
        # Robot is ready if .init is True (after self.robot.initialize())
        # Cameras are ready if realsense.is_ready is True (SingleRealsense sets ready_event).
        return self.realsense.is_ready and self.robot.init

    def start(self, wait=True):
        logger.info("RealEnv.start() called.")
        # Start cameras (non-blocking)
        self.realsense.start(wait=False)
        # Start visualizer if any
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        # Initialize xArm
        self.robot.initialize()
        if self.init_joints:
            self.robot.home()

        if wait:
            logger.info("RealEnv.start_wait() called.")
            self.start_wait()
        logger.info(f"RealEnv start done. is_ready={self.is_ready}")

    def stop(self, wait=True):
        logger.info("RealEnv.stop() called.")
        self.end_episode()  # in case we were recording
        # Stop visualizer
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        # Stop cameras
        self.realsense.stop(wait=False)
        # Shutdown xArm
        self.robot.shutdown()

        if wait:
            logger.info("RealEnv.stop_wait() called.")
            self.stop_wait()
        logger.info("RealEnv stop done.")

    def start_wait(self):
        logger.info("RealEnv.start_wait() -> realsense.start_wait()")
        self.realsense.start_wait()
        if self.multi_cam_vis is not None:
            logger.info("RealEnv.start_wait() -> multi_cam_vis.start_wait()")
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        logger.info("RealEnv.stop_wait() -> realsense.stop_wait()")
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            logger.info("RealEnv.stop_wait() -> multi_cam_vis.stop_wait()")
            self.multi_cam_vis.stop_wait()

    def __enter__(self):
        logger.info("Entering RealEnv context manager.")
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting RealEnv context manager.")
        self.stop()

    def get_obs(self) -> dict:

        if not self.is_ready:
            logger.warning("RealEnv.get_obs() called but system not ready!")
            return {}

        k = max(self.n_obs_steps, 1)
        # Attempt to get k frames from each camera
        logger.debug(f"RealEnv.get_obs() retrieving {k} frames from realsense.")
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # Robot
        code, pose = self.robot.arm.get_position()
        xarm_pose = np.array(pose[:6], dtype=np.float32)
        xarm_gripper = np.array([self.robot.previous_grasp], dtype=np.float32)

        # We'll replicate the single robot reading n_obs_steps times
        stacked_pose = np.tile(xarm_pose[None, :], (self.n_obs_steps, 1))
        stacked_gripper = np.tile(xarm_gripper[None, :], (self.n_obs_steps, 1))

        # Build camera obs
        camera_obs = {}
        for idx, val in self.last_realsense_data.items():
            camera_obs[f"camera_{idx}"] = val["color"][-self.n_obs_steps:]

        obs_data = dict(camera_obs)
        obs_data["robot_eef_pose"] = stacked_pose
        obs_data["robot_gripper"] = stacked_gripper
        # simple timestamps
        now = time.time()
        obs_data["timestamp"] = np.linspace(
            now - (self.n_obs_steps - 1) * (1.0 / self.frequency),
            now,
            self.n_obs_steps,
        )

        # If accumulators exist, store 1-step
        if self.obs_accumulator is not None:
            now_array = np.array([now], dtype=np.float64)
            raw_obs = {
                "xarm_eef_pose": xarm_pose[None, :],
                "xarm_gripper": xarm_gripper[None, :],
            }
            self.obs_accumulator.put(raw_obs, now_array)

        return obs_data

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray):
        if not self.is_ready:
            logger.warning("RealEnv.exec_actions() called but system not ready!")
            return

        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        if len(actions) > 0:
            action = actions[0]
            dpos = action[:3]
            drot = action[3:6] if len(action) >= 6 else np.zeros(3)
            grasp = action[6] if len(action) >= 7 else 0.0
            self.robot.step(dpos, drot, grasp)

        # Record if accumulators exist
        if self.action_accumulator is not None:
            self.action_accumulator.put(actions, timestamps)

    def get_robot_state(self):

        code, pose = self.robot.arm.get_position()
        xarm_pose = np.array(pose[:6], dtype=np.float32)
        gripper = self.robot.previous_grasp
        return {"eef_pose": xarm_pose, "gripper": gripper}

    def start_episode(self, start_time=None):
        if not self.is_ready:
            logger.warning("Cannot start_episode() - system not ready.")
            return

        if start_time is None:
            start_time = time.time()
        self.start_time = start_time
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)

        # Build paths for n cameras
        n_cams = self.realsense.n_cameras
        video_paths = []
        for i in range(n_cams):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1.0 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1.0 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1.0 / self.frequency
        )
        logger.info(f"Episode {episode_id} started at t={start_time:.3f}.")

    def end_episode(self):
        if not self.is_ready:
            logger.warning("end_episode() called but system not ready or no episode in progress.")
            return

        self.realsense.stop_recording()
        if self.obs_accumulator is None:
            # no episode actually started
            return

        obs_data = self.obs_accumulator.data
        obs_timestamps = self.obs_accumulator.timestamps
        actions = self.action_accumulator.actions
        action_timestamps = self.action_accumulator.timestamps
        stages = self.stage_accumulator.actions

        n_steps = min(len(obs_timestamps), len(action_timestamps))
        if n_steps > 0:
            episode = {}
            episode["timestamp"] = obs_timestamps[:n_steps]
            episode["action"] = actions[:n_steps]
            episode["stage"] = stages[:n_steps]
            for k, arr in obs_data.items():
                episode[k] = arr[:n_steps]
            self.replay_buffer.add_episode(episode, compressors="disk")
            ep_id = self.replay_buffer.n_episodes - 1
            logger.info(f"Episode {ep_id} saved with {n_steps} steps.")
        else:
            logger.info("No steps recorded in this episode.")

        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        ep_id = self.replay_buffer.n_episodes
        vid_dir = self.video_dir.joinpath(str(ep_id))
        if vid_dir.exists():
            shutil.rmtree(str(vid_dir))
        logger.info(f"Episode {ep_id} dropped!")
"""
