import logging
import pathlib

from multiprocessing.managers import SharedMemoryManager
from ril_env.xarm_controller import XArmConfig, XArmController
from ril_env.replay_buffer import ReplayBuffer
from ril_env.realsense import SingleRealsense
from ril_env.cv2_util import get_image_transform, optimal_row_cols
from ril_env.video_recorder import VideoRecorder
from ril_env.multi_realsense import MultiRealsense
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from typing import Tuple, List, Optional, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OBS_KEY_MAP = {
    # Robot
    "ActualTCPPose": "robot_eef_pose",
    "ActualTCPSpeed": "robot_eef_pose_vel",
    "ActualQ": "robot_joint",
    "ActualQd": "robot_joint_vel",
    # Timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class RealEnv:
    def __init__(
        self,
        output_dir: Union[pathlib.Path, str] = "./recordings/",
        xarm_config: Optional[XArmConfig] = None,
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
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')

        logger.info(f"[RealEnv] Output directory: {output_dir}")
        logger.info(f"[RealEnv] Video directory: {video_dir}")
        logger.info(f"[RealEnv] Replay buffer path: {zarr_path}")

        if xarm_config is None:
            xarm_config = XArmConfig()

        assert xarm_config.frequency <= video_capture_fps, "Cannot run frequency faster than video capture."

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
            recording_fps = xarm_config.frequency
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
            verbose=True,
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
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = xarm_config.frequency
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

        self.start_time

    # start-stop API
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready
