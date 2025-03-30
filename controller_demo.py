import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import logging

from ril_env.spacemouse import Spacemouse
from ril_env.xarm_controller import XArm, XArmConfig, XArmController
from ril_env.multi_realsense import MultiRealsense
from ril_env.video_recorder import VideoRecorder

logger = logging.getLogger(__name__)

def main():
    xarm_config = XArmConfig()

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
         XArm(xarm_config) as arm, \
         MultiRealsense(
             shm_manager=shm_manager,
             record_fps=30,
             capture_fps=30,
             put_fps=30,
             put_downsample=True,
             video_recorder=video_recorder,
             verbose=True,
         ) as multi:

        # Start the XArmController for data collection.
        xarm_ctrl = XArmController(arm, shm_manager, frequency=50, verbose=True)
        xarm_ctrl.start()
        
        print("MultiRealsense cameras started and ready.")
        multi.set_exposure(exposure=120, gain=0)
        multi.set_white_balance(white_balance=5900)

        multi.start_recording(str(video_dir), start_time=time.time())
        print(f"Recording started. Videos will be saved in {video_dir.absolute()}.")

        try:
            while True:
                loop_start = time.monotonic()
                time.sleep(0.01)

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                # If the designated button is pressed, re-home the arm.
                if sm.is_button_pressed(1):
                    arm.home()
                    continue

                # Command a step via the controller.
                xarm_ctrl.step(dpos, drot, grasp)

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, 0.02 - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            multi.stop_recording()
            print("Recording stopped.")

            # Stop the controller process and wait for it to finish.
            xarm_ctrl.stop()
            xarm_ctrl.join()

            # Retrieve collected data.
            collected_data = xarm_ctrl.get_state()
            num_samples = len(collected_data)
            print(f"Collected {num_samples} entries from the ring buffer:")
            if num_samples == 0:
                print("No data was collected!")
            else:
                # If there are fewer than 10 entries, print all of them.
                samples_to_print = collected_data if num_samples < 10 else xarm_ctrl.get_state(k=10)
                for sample in samples_to_print:
                    print(sample)

if __name__ == "__main__":
    main()
