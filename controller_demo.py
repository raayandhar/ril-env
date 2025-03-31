import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import logging

# Suppose these come from your own modules:
from ril_env.spacemouse import Spacemouse
from ril_env.multi_realsense import MultiRealsense
from ril_env.video_recorder import VideoRecorder

# Import the new xarm classes
from ril_env.xarm_controller import XArmConfig, XArmProcess, XArmClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 1) xArm config
    xarm_config = XArmConfig(
        ip="192.168.1.223",
        tcp_maxacc=5000,
        position_gain=2.0,
        orientation_gain=2.0,
        home_pos=[0, 0, 0, 70, 0, 70, 0],
        home_speed=50.0,
        verbose=True,
    )

    # 2) Video setup
    video_dir = pathlib.Path("recordings")
    video_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = VideoRecorder.create_h264(
        fps=30,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=21,
        thread_type="FRAME",
        thread_count=3,
    )

    with SharedMemoryManager() as shm_manager, \
         Spacemouse(deadzone=0.4) as sm, \
         MultiRealsense(
             shm_manager=shm_manager,
             record_fps=30,
             capture_fps=30,
             put_fps=30,
             put_downsample=True,
             video_recorder=video_recorder,
             verbose=True,
         ) as multi:

        # 3) Create the XArmProcess in the main process
        xarm_proc = XArmProcess(
            config=xarm_config,
            shm_manager=shm_manager,
            frequency=20.0,   # 20 Hz data capture
            buffer_size=256,  # ring buffer capacity
        )

        # Wrap it with XArmClient
        xarm_client = XArmClient(xarm_proc)

        # Start the xarm background process
        xarm_client.start(wait=2.0)

        logger.info("MultiRealsense cameras started and ready.")
        multi.set_exposure(exposure=120, gain=0)
        multi.set_white_balance(white_balance=5900)

        # Start recording
        multi.start_recording(str(video_dir), start_time=time.time())
        logger.info(f"Recording started. Videos will be saved in {video_dir.absolute()}.")

        try:
            while True:
                loop_start = time.monotonic()

                # 1) Grab spacemouse motion
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp  # 0 or 1

                # If button 1 pressed => home
                if sm.is_button_pressed(1):
                    xarm_client.home()
                    continue

                # 2) Send step command
                xarm_client.step(dpos, drot, grasp)

                # 3) Retrieve the latest ring-buffer sample
                latest_sample = xarm_client.get_state(k=None)
                # For printing demonstration:
                # If k=None, we get a single sample dict => "TCPPose", etc.
                # If k=10, we'd get the last 10 samples as stacked arrays.

                if latest_sample:
                    print("Latest xArm data:")
                    for key, val in latest_sample.items():
                        print(f"  {key}: {val}")

                # Basic timing
                elapsed = time.monotonic() - loop_start
                # We'll do ~50 Hz loop here
                time.sleep(max(0, 0.02 - elapsed))

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            multi.stop_recording()
            logger.info("Recording stopped.")

            # 4) Stop xarm background process
            xarm_client.stop()

            # Dump all data in ring buffer if you like:
            # full_data = xarm_client.get_state(k=256)  # last 256 samples
            # or xarm_client.proc.ring_buffer.get_all()
            # do something with that data if needed


if __name__ == "__main__":
    main()
