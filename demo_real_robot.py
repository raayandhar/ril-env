#!/usr/bin/env python

"""
Usage:
(venv)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_xarm>

Robot movement (with SpaceMouse):
- Default: Move the robot EEF in XY plane by moving the SpaceMouse.
- Press left button on the SpaceMouse to enable rotation control (and lock translation).
- Press right button on the SpaceMouse to unlock z-axis translation.

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
import logging
import pathlib

from multiprocessing.managers import SharedMemoryManager

from ril_env.spacemouse import Spacemouse
from ril_env.keystroke_counter import KeystrokeCounter, Key, KeyCode
from ril_env.precise_sleep import precise_wait
from ril_env.xarm_controller import XArmConfig
from ril_env.real_env import RealEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output", "-o", default='./recordings/', required=True, help="Directory to save demonstration dataset."
)
@click.option(
    "--robot_ip", "-ri", default="192.168.1.223", required=True, help="xArm's IP address, e.g. 192.168.1.223"
)
@click.option(
    "--vis_camera_idx",
    default=0,
    type=int,
    help="Which RealSense camera index to visualize in the OpenCV window.",
)
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=False,
    help="Whether to home the robot on startup.",
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.01,
    type=float,
    help="Latency between receiving SpaceMouse command and sending it to the robot, in seconds.",
)
@click.option(
    "--record_res",
    default=(1280, 720),
    type=(int, int),
    help="Resolution for recording, format: WIDTH HEIGHT"
)
@click.option(
    "--spacemouse_deadzone",
    default=0.05,
    type=float,
    help="Deadzone for the SpaceMouse input",
)
@click.option(
    "--position_gain",
    default=5.0,
    type=float,
    help="Scaling factor for position control",
)
@click.option(
    "--orientation_gain",
    default=10.0,
    type=float,
    help="Scaling factor for orientation control",
)
def main(
    output, 
    robot_ip, 
    vis_camera_idx, 
    init_joints, 
    frequency, 
    command_latency,
    record_res,
    spacemouse_deadzone,
    position_gain,
    orientation_gain
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xarm_config = XArmConfig(robot_ip=robot_ip)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=spacemouse_deadzone,
            shm_manager=shm_manager
        ) as sm, RealEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=2,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            init_joints=init_joints,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1280, 720),
            shm_manager=shm_manager,
        ) as env:
            print("\n\n HERE1!! \n\n")
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)
            print("\n\n HERE2!! \n\n")
            time.sleep(1)
            logger.info("System initialized")
            print("\n\n HERE3!! \n\n")
            state = env.get_robot_state()
            target_pose = np.array(state["TCPPose"], dtype=np.float32)
            logger.info(f"Initial pose: {target_pose}")
            print("\n\n HERE4!! \n\n")
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            print("\n\n HERE5!! \n\n")
            try:
                logger.info("Starting main control loop...")
                while not stop:
                    print("\n\n HERE6!! \n\n")
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + dt
                    t_sample = t_cycle_end - command_latency
                    print("\n\n HERE7!! \n\n")
                    obs = env.get_obs()
                    print("\n\n HERE8!! \n\n")
                    press_events = key_counter.get_press_events()
                    print("\n\n HERE9!! \n\n")
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            # Exit program
                            logger.info("Quit requested...")
                            stop = True
                        elif key_stroke == KeyCode(char="c"):
                            # Start recording
                            env.start_episode()
                            is_recording = True
                            logger.info("Recording started!")
                        elif key_stroke == KeyCode(char="s"):
                            # Stop recording
                            env.end_episode()
                            is_recording = False
                            logger.info("Recording stopped.")
                        elif key_stroke == Key.backspace:
                            # Delete the most recent recorded episode
                            if click.confirm("Drop the most recently recorded episode?"):
                                env.drop_episode()
                                is_recording = False
                                logger.info("Episode dropped.")
                    print("\n\n HERE10!! \n\n")
                    # Check for stage information (e.g., spacebar pressed)
                    stage_val = key_counter[Key.space]  # 0 if not pressed, 1 if pressed
                    print("\n\n HERE11!! \n\n")
                    # Visualize camera feed
                    vis_img = obs[f"camera_{vis_camera_idx}"][-1, :, :, ::-1].copy()
                    print("\n\n HERE12!! \n\n")
                    # Add text overlay with status information
                    episode_id = env.replay_buffer.n_episodes
                    text_str = f"Episode: {episode_id}, Stage: {stage_val}"
                    if is_recording:
                        text_str += " [RECORDING]"
                    print("\n\n HERE13!! \n\n")
                    cv2.putText(
                        vis_img,
                        text_str,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=2,
                    )
                    print("\n\n HERE14!! \n\n")
                    #cv2.imshow("Robot Teleop", vis_img)
                    #cv2.waitKey(1)
                    print("\n\n HERE15!! \n\n")
                    # Wait precisely until sample time, then read SpaceMouse
                    precise_wait(t_sample)
                    print("\n\n HERE16!! \n\n")
                    sm_state = sm.get_motion_state_transformed()
                    print(f"SM state: {sm_state}")
                    dpos = sm_state[:3]
                    drot = sm_state[3:]
                    grasp = sm.grasp

                    dpos *= xarm_config.position_gain
                    drot *= xarm_config.orientation_gain

                    curr_rot = st.Rotation.from_euler("xyz", target_pose[3:], degrees=True)
                    delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                    final_rot = delta_rot * curr_rot
                    
                    target_pose[:3] += dpos
                    target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                    action = np.concatenate([target_pose, [grasp]])

                    exec_timestamp = t_command_target - time.monotonic() + time.time()
                    env.exec_actions(
                        actions=[action],
                        timestamps=[exec_timestamp],
                        stages=[stage_val],
                    )

                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user.")
            except Exception:
                logger.error("Exception occurred during the main loop:")
                traceback.print_exc()
            finally:
                logger.info("Exiting main loop. Cleaning up...")
                # Stop recording if still active
                if is_recording:
                    try:
                        env.end_episode()
                        logger.info("Recording stopped during cleanup.")
                    except:
                        logger.warning("Failed to cleanly stop recording.")
                
                # Close OpenCV windows
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
