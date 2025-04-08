import logging
import time
import math
import numpy as np
import shutil
import pathlib

from multiprocessing.managers import SharedMemoryManager
from ril_env.xarm_controller import XArmConfig, XArmController
from ril_env.replay_buffer import ReplayBuffer
from ril_env.realsense import SingleRealsense
from ril_env.cv2_util import get_image_transform, optimal_row_cols
from ril_env.video_recorder import VideoRecorder
from ril_env.multi_realsense import MultiRealsense
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
)
from typing import Tuple, List, Optional, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OBS_KEY_MAP = {
    # Robot - using the correct key names that match XArmController.get_all_state()
    "TCPPose": "robot_eef_pose",
    "TCPSpeed": "robot_eef_pose_vel",
    "JointAngles": "robot_joint",
    "JointSpeeds": "robot_joint_vel",
    # Additional keys if they exist
    "Grasp": "robot_gripper",
    "robot_receive_timestamp": "robot_timestamp",
    # Timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class RealEnv:
    def __init__(
        self,
        output_dir: Union[pathlib.Path, str] = "./recordings/",
        xarm_config: Optional[XArmConfig] = None,
        frequency: int = 30,
        num_obs_steps: int = 2,
        obs_image_resolution: Tuple[int, int] = (640, 480),
        max_obs_buffer_size: int = 30,
        camera_serial_numbers: Optional[List[int]] = None,
        obs_key_map: Dict = DEFAULT_OBS_KEY_MAP,
        obs_float32: bool = False,
        init_joints: bool = False,
        video_capture_fps: int = 30,
        video_capture_resolution: Tuple[int, int] = (1280, 720),
        record_raw_video: bool = True,
        thread_per_video: int = 3,
        video_crf: int = 3,
        enable_multi_cam_vis: bool = False,
        multi_cam_vis_resolution: Tuple[int, int] = (1280, 720),
        shm_manager: Optional[SharedMemoryManager] = None,
    ):
        logger.info("[RealEnv] Initializing environment.")

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        logger.info(f"[RealEnv] Output directory: {output_dir}")
        logger.info(f"[RealEnv] Video directory: {video_dir}")
        logger.info(f"[RealEnv] Replay buffer path: {zarr_path}")

        if xarm_config is None:
            xarm_config = XArmConfig()

        assert (
            frequency <= video_capture_fps
        ), "Cannot run frequency faster than video capture."

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            logger.info("[RealEnv] Started local SharedMemoryManager")
        self.shm_manager = shm_manager

        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        logger.info(f"[RealEnv] Camera serial numbers: {camera_serial_numbers}")

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

        # Multi-cam visual transformation
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
        logger.info(f"[RealEnv] Recording FPS: {recording_fps}")

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            record_fps=recording_fps,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=True,
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

        robot = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )

        self.realsense = realsense
        self.robot = robot
        self.xarm_config = xarm_config
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.num_obs_steps = num_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map

        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.last_realsense_data = None

        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
        
        # Count accumulated observations for debugging
        self.obs_count = 0

    # Start-stop API
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        if wait:
            self.realsense.start(wait=True)
            self.robot.start(wait=True)
        else:
            self.realsense.start(wait=False)
            self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)

    def stop(self, wait=True):
        try:
            # Only try to end the episode if there's an active one
            if self.obs_accumulator is not None:
                self.end_episode()
        except Exception as e:
            logger.error(f"Error in end_episode during stop: {e}")
            
        if wait:
            self.realsense.stop(wait=True)
            self.robot.stop(wait=True)
        else:
            self.realsense.stop(wait=False)
            self.robot.stop(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # Async env API
    def get_obs(self) -> Dict:
        "observation dict"
        assert self.is_ready

        k = math.ceil(self.num_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # Running at 50 hz
        last_robot_data = self.robot.get_all_state()
        # Uncomment for debugging
        # print(f"ROBOT STATE KEYS: {list(last_robot_data.keys())}")

        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (
            np.arange(self.num_obs_steps)[::-1] * dt
        )

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]

        robot_timestamps = last_robot_data["robot_receive_timestamp"]
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # Accumulate observations for recording
        if self.obs_accumulator is not None:
            # The TimestampObsAccumulator expects each value to be a single observation,
            # not the array for multiple steps. We need to extract just the latest observation.
            latest_obs = {}
            for k, v in last_robot_data.items():
                if k in self.obs_key_map:
                    mapped_key = self.obs_key_map[k]
                    
                    # Make sure we're passing a single observation
                    if isinstance(v, np.ndarray) and len(v.shape) > 0:
                        latest_value = v[-1]  # Take the most recent value
                        
                        # Handle scalar values correctly
                        if np.isscalar(latest_value):
                            latest_obs[mapped_key] = np.array([latest_value])
                        else:
                            latest_obs[mapped_key] = latest_value
                    else:
                        latest_obs[mapped_key] = v
            
            # Now accumulate the observation with correct format
            if latest_obs:  # Only if we have any observations
                self.obs_count += 1
                if self.obs_count % 10 == 0:  # Log every 10th observation to reduce spam
                    logger.info(f"Accumulated {self.obs_count} observations, latest keys: {list(latest_obs.keys())}")
                self.obs_accumulator.put(latest_obs, np.array([robot_timestamps[-1]]))

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data["timestamp"] = obs_align_timestamps
        return obs_data

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        stages: Optional[np.ndarray] = None,
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        for i in range(len(new_actions)):
            new_action = new_actions[i]
            pose = new_action[:6]
            grasp = new_action[-1]
            # Should we have a target timestamp?
            self.robot.step(pose, grasp)

        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps,
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps,
            )

    def get_robot_state(self):
        return self.robot.get_state()

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        
        # Reset observation counter
        self.obs_count = 0
        
        logger.info(f"Episode {episode_id} started!")

    def end_episode(self):
        """Safely end the current episode and save data to the replay buffer."""
        if not self.is_ready:
            logger.warning("Tried to end episode but environment is not ready")
            return
            
        # Stop recording video
        self.realsense.stop_recording()

        # If we don't have any accumulators, there's nothing to save
        if self.obs_accumulator is None or self.action_accumulator is None or self.stage_accumulator is None:
            logger.info("No active episode to end or episode already ended")
            return

        try:
            # Get robot state for final observation
            last_robot_data = self.robot.get_all_state()
            
            # Debug: print what attributes are in the accumulators
            print(f"Action accumulator dir: {dir(self.action_accumulator)}")
            print(f"Stage accumulator dir: {dir(self.stage_accumulator)}")
            
            # Get action data
            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            
            # Try different ways to access stages
            try:
                stages = self.stage_accumulator.actions
                print(f"Using stage_accumulator.actions: shape={stages.shape}")
            except Exception as e:
                print(f"Error accessing stage_accumulator.actions: {e}")
                # Fallback
                stages = np.zeros_like(actions[:, 0], dtype=np.int64)
            
            # Check if we have actions to save
            n_steps = len(action_timestamps)
            logger.info(f"Actions to save: {n_steps}")
            
            if n_steps > 0:
                # Create episode dictionary
                episode = dict()
                episode["timestamp"] = action_timestamps
                episode["action"] = actions
                episode["stage"] = stages
                
                # Construct observation data matching the number of actions
                # Map robot state to the right keys
                robot_obs = {}
                for k, v in last_robot_data.items():
                    if k in self.obs_key_map:
                        mapped_key = self.obs_key_map[k]
                        
                        # Get the latest value
                        if isinstance(v, np.ndarray) and len(v.shape) > 0:
                            latest_value = v[-1]
                        else:
                            latest_value = v
                        
                        # Replicate it for all timesteps
                        if np.isscalar(latest_value):
                            robot_obs[mapped_key] = np.full(n_steps, latest_value)
                        else:
                            # For arrays like pose data
                            robot_obs[mapped_key] = np.tile(latest_value, (n_steps, 1))
                
                # Add robot observations to episode
                for key, value in robot_obs.items():
                    episode[key] = value
                
                # Log episode data before saving
                logger.info(f"Episode data keys: {list(episode.keys())}")
                logger.info(f"Episode timestamp shape: {episode['timestamp'].shape}")
                logger.info(f"Episode action shape: {episode['action'].shape}")
                
                # Save to replay buffer
                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                logger.info(f"Episode {episode_id} saved with {n_steps} steps!")
                
                # Print a confirmation for the user
                print(f"Data successfully saved to {self.output_dir}/replay_buffer.zarr!")
                print(f"To view your data, you need to load it with the ReplayBuffer class.")
                print("Example:")
                print("  from ril_env.replay_buffer import ReplayBuffer")
                print(f"  buffer = ReplayBuffer.create_from_path('{self.output_dir}/replay_buffer.zarr')")
                print("  episode = buffer.get_episode(0)  # Get the first episode")
                print("  print(episode.keys())  # See what data is available")
                print("  print(episode['action'].shape)  # Check the action shape")
            else:
                logger.warning("No steps to save in this episode")
        except Exception as e:
            logger.error(f"Error saving episode data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always clear the accumulators, even if there was an error
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {episode_id} dropped!")
        
    def verify_data(self):
        """Verify that data is saved in the replay buffer and print some statistics."""
        if self.replay_buffer.n_episodes == 0:
            print("No episodes found in the replay buffer.")
            return
            
        print(f"Found {self.replay_buffer.n_episodes} episodes in the replay buffer.")
        
        # Print information about the first episode
        episode = self.replay_buffer.get_episode(0)
        print(f"Episode 0 data keys: {list(episode.keys())}")
        for key, value in episode.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if len(value) > 0:
                    if len(value.shape) == 1:
                        print(f"    First few values: {value[:5]}")
                    else:
                        print(f"    First value: {value[0]}")
