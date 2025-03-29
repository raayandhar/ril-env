#!/usr/bin/env python

"""
Usage:
(venv)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_xarm>

Robot movement (with SpaceMouse):
- Default: Move the robot EEF in XY plane by moving the SpaceMouse.
- Press left button on the SpaceMouse to enable rotation control (and lock translation).
- Press right button on the SpaceMouse to unlock z-axis translation.
  (This is just an example; adapt the logic as desired in the code.)

Recording control:
- Click the OpenCV window (ensure it is in focus).
- Press "C" to start recording an episode.
- Press "S" to stop recording.
- Press "Q" to exit the program.
- Press "Backspace" to delete the previously recorded episode.
"""

import time
import traceback
import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from multiprocessing.managers import SharedMemoryManager

from ril_env.spacemouse import Spacemouse
from ril_env.keystroke_counter import KeystrokeCounter, Key, KeyCode
from ril_env.precise_sleep import precise_wait

# Import the environment we created:
from ril_env.real_env import RealEnv


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="xArm's IP address, e.g. 192.168.1.223")
@click.option('--vis_camera_idx', default=0, type=int,
              help="Which RealSense camera index to visualize in the OpenCV window.")
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Whether to home the robot on startup.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float,
              help="Latency between receiving SpaceMouse command and sending it to the robot, in seconds.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    """
    An example teleoperation + recording script for a real xArm, controlled by a SpaceMouse.
    Cameras are from MultiRealsense, video is saved upon 'C' / 'S' keystrokes, and episodes
    are stored in replay_buffer.zarr plus .mp4 files.
    """
    dt = 1.0 / frequency

    with SharedMemoryManager() as shm_manager:
        # Example "KeystrokeCounter" to gather pressed keys, e.g. 'c', 's', 'q', etc.
        with KeystrokeCounter() as key_counter, \
             Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                init_joints=init_joints,
                # For demonstration, let's record at raw resolution and enable the multi-cam visualizer
                obs_image_resolution=(1280, 720),
                video_capture_resolution=(1280, 720),
                enable_multi_cam_vis=False,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
             ) as env:
            print("Here 75")
            # Optionally configure camera settings:
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)

            print("===== Ready! =====")
            # We'll pick an initial 'target_pose' from the current robot state.
            state = env.get_robot_state()
            target_pose = np.array(state['eef_pose'], dtype=np.float32)

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            try:
                print("Entering main loop...")
                while not stop:
                    print("here2")
                    # A cycle from [t_start + iter_idx * dt, t_start + (iter_idx+1)*dt]
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    # We'll apply commands in the *next* cycle:
                    t_command_target = t_cycle_end + dt
                    # We might sample the environment a bit earlier, to simulate command latency:
                    t_sample = t_cycle_end - command_latency

                    # Acquire current observation(s) at ~10Hz
                    obs = env.get_obs()

                    # Check key presses
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Press 'Q' => exit the program
                            stop = True
                        elif key_stroke == KeyCode(char='c'):
                            # Press 'C' => start new episode recording
                            env.start_episode()
                            is_recording = True
                            print("[demo] Recording started!")
                        elif key_stroke == KeyCode(char='s'):
                            # Press 'S' => stop recording
                            env.end_episode()
                            is_recording = False
                            print("[demo] Recording stopped.")
                        elif key_stroke == Key.backspace:
                            # Press 'Backspace' => drop the last recorded episode
                            if click.confirm("Drop the most recently recorded episode?"):
                                env.drop_episode()
                                is_recording = False

                    # If you'd like to store a discrete "stage" (like "1" while space is pressed):
                    stage_val = key_counter[Key.space]  # e.g. 0 if not pressed, or 1 if pressed

                    # Visualize the specified camera index.
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()

                    # Text overlay
                    episode_id = env.replay_buffer.n_episodes
                    text_str = f"Episode: {episode_id}, Stage: {stage_val}"
                    if is_recording:
                        text_str += " [REC]"
                    cv2.putText(
                        vis_img,
                        text_str,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=2,
                    )
                    cv2.imshow("teleop_view", vis_img)
                    cv2.pollKey()

                    # Wait precisely until t_sample, then read the SpaceMouse
                    precise_wait(t_sample)
                    sm_state = sm.get_motion_state_transformed()  # shape (6,) => [dx,dy,dz, rx,ry,rz]
                    # Scale the commands based on robot max speeds:
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)

                    # Mode control based on button presses:
                    if not sm.is_button_pressed(0):
                        drot_xyz[:] = 0
                    if not sm.is_button_pressed(1):
                        dpos[2] = 0

                    # Combine rotation with current target orientation:
                    curr_rot = st.Rotation.from_euler('xyz', target_pose[3:], degrees=True)
                    delta_rot = st.Rotation.from_euler('xyz', drot_xyz, degrees=True)
                    final_rot = delta_rot * curr_rot

                    # Update target_pose
                    target_pose[:3] += dpos
                    target_pose[3:] = final_rot.as_euler('xyz', degrees=True)

                    # Execute actions, including stage information:
                    env.exec_actions(
                        actions=[target_pose],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        stages=[stage_val],
                    )

                    # Wait until the cycle is complete
                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except Exception as e:
                print("Exception occurred during the main loop:")
                traceback.print_exc()
            finally:
                print("Exiting main loop. Performing cleanup if necessary.")
                # Additional cleanup can be added here if needed.


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
