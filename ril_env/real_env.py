import pathlib
import numpy as np
import logging

from multiprocessing.managers import SharedMemoryManager

from ril_env.xarm_controller import XArm, XArmConfig
from ril_env.multi_realsense import MultiRealsense, SingleRealsense
from ril_env.video_recorder import VideoRecorder
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.replay_buffer import ReplayBuffer
from ril_env.cv2_util import get_image_transform, optimal_row_cols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Missing DEFAULT_OBS_KEY_MAP
# Should write a method "get_state" where we can just get this
# from the robot.
# What does this do?
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
# Need to write a robot controller interface to pick up actions
# from the ring buffer.
# See: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/rtde_interpolation_controller.py#L211


class RealEnv:
    def __init__(
        self,
        # Required?
        output_dir="./recordings/",
        # robot_ip,
        # Environment parameters
        frequency=30,  # Robot control frequency
        num_obs_steps=2,  # Used in the async env API
        # Observation parameters
        obs_image_resolution=(640, 480),
        max_obs_buffer_size=30,
        camera_serial_numbers=None,
        obs_key_map=DEFAULT_OBS_KEY_MAP,  # REQUIRES CHANGING
        obs_float32=False,
        # Action - are these necessary / values make sense?
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        # Robot - again, necessary / requires changing?
        tcp_offset=0.13,  # ?? Not necessary for our robot code
        init_joints=False,  # This should be homing
        # Video capture - see demo.py for what's needed?
        video_capture_fps=30,
        video_capture_resolution=(1280, 720),
        # Saving params
        record_raw_video=True,
        thread_per_video=3,
        video_crf=21,
        # Vis params
        enable_multi_cam_vis=False,  # Not gonna work on this for now
        multi_cam_vis_resolution=(1280, 720),
        shm_manager=None,
    ):

        logger.info("Initializing environment.")

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

        # Video recorder
        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,  # 3
        )

        # Up to this point looks fine, video recorder is good
        # Now the difficult portion: realsense
        # Should test this with demo.py to check if settings are fine
        logger.info(f"Recording FPS: {recording_fps}")
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

        robot_config = XArmConfig()
        self.robot = XArm(robot_config)

        # Store
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.realsense = realsense
        self.multi_cam_vis = multi_cam_vis

        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.num_obs_steps = num_obs_steps
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
        # Init works.

    @property
    def is_ready(self):
        # Realsense is not passing!
        logger.info(f"Realsense ready? {self.realsense.is_ready}")
        logger.info(f"Robot ready? {self.robot.is_ready}")
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        logger.info("RealEnv.start() called")
        self.realsense.start(wait=True)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=True)
        self.robot.initialize()
        if self.init_joints:
            self.robot.home()

        logger.info(f"RealEnv start done. is_ready={self.is_ready}")

    def stop(self, wait=True):
        logger.info("RealEnv.stop() called (no wait).")
        # self.end_episode()
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


def main():
    with RealEnv() as real_env:
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
