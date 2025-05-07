import time
import traceback
import click
import numpy as np
import scipy.spatial.transform as st
import logging
import pathlib
import zarr

from multiprocessing.managers import SharedMemoryManager

from ril_env.control.spacemouse import Spacemouse
from ril_env.control.xarm_controller import XArmConfig, Command
from ril_env.utils.keystroke_counter import KeystrokeCounter, Key, KeyCode
from ril_env.utils.precise_sleep import precise_wait
from ril_env.rilenv import RILEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
TODO:
- Actually check that we are recording things
- Fix the camera visualization
- Fix the spacemouse-robot weird movement problem
- Clean up the code, add some documentation
- Demonstrations!
"""


def main(
    output="./recordings/",
    init_joints=True,  # Not used ATM
    frequency=30,  # Cannot increase frequency
    command_latency=0.01,
    record_res=(1280, 720),
    spacemouse_deadzone=0.05,
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    xarm_config = XArmConfig()

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=spacemouse_deadzone, shm_manager=shm_manager
        ) as sm, RILEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=2,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=3000)

            time.sleep(1)
            logger.info("System initialized")

            state = env.get_robot_state()
            target_pose = np.array(state["TCPPose"], dtype=np.float32)
            logger.info(f"Initial pose: {target_pose}")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            def move_to_home():
                """Move the robot to home position"""
                nonlocal target_pose
                logger.info("Moving to home position...")
                
                # First, clean any errors and set the correct mode for homing
                clean_cmd = {
                    "cmd": Command.STEP.value,  # Using STEP command as a vehicle to send a message
                    "target_pose": target_pose,  # Keep current pose
                    "grasp": 0.0,
                    "duration": 0.0,
                    "target_time": time.time(),
                    "clean_errors": True,  # Custom field to indicate cleaning errors
                }
                env.robot.input_queue.put(clean_cmd)
                time.sleep(0.5)  # Give time for error cleaning
                
                # Now send the actual home command
                command = {
                    "cmd": Command.HOME.value,
                    "target_pose": np.zeros(6, dtype=np.float64),
                    "grasp": 0.0,
                    "duration": 0.0,
                    "target_time": time.time(),
                }
                env.robot.input_queue.put(command)
                
                # Give more time for homing to complete
                time.sleep(3.0)
                
                state = env.get_robot_state()
                target_pose = np.array(state["TCPPose"], dtype=np.float32)
                logger.info(f"Robot homed. New pose: {target_pose}")
                
                # Send an explicit action at the homed position
                exec_timestamp = time.time()
                action = np.concatenate([target_pose, [0.0]])
                env.exec_actions(
                    actions=[action],
                    timestamps=[exec_timestamp],
                    stages=[0],
                )
                time.sleep(0.2)
                
                # Force a complete reset of episode handling
                # End any current episode
                if hasattr(env, 'obs_accumulator') and env.obs_accumulator is not None:
                    env.end_episode()
                    time.sleep(0.2)
                    
                # Explicitly reset any accumulators
                env.obs_accumulator = None
                env.action_accumulator = None
                env.stage_accumulator = None
                
                # Get fresh observations to reset internal buffers
                obs = env.get_obs()
                time.sleep(0.3)  # Increased delay for more stability
                
            def replay_last_episode():
                """Home the robot and replay the most recent recording"""
                # First move to home position
                move_to_home()
                logger.info("Preparing to replay most recent episode...")
                
                # Open the replay buffer
                zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
                try:
                    replay_buffer = zarr.open(zarr_path, mode="r")
                    
                    # Get episode information
                    episode_ends = replay_buffer["meta"]["episode_ends"][:]
                    total_episodes = len(episode_ends)
                    
                    if total_episodes == 0:
                        logger.error("No episodes found in the replay buffer")
                        return
                    
                    # Get the most recent episode (last one)
                    episode_idx = total_episodes - 1
                    
                    # Calculate episode boundaries
                    start_idx = 0
                    if episode_idx > 0:
                        start_idx = episode_ends[episode_idx - 1]
                    end_idx = episode_ends[episode_idx]
                    
                    # Check if episode is valid
                    episode_length = end_idx - start_idx
                    if episode_length <= 0:
                        logger.error(f"Episode {episode_idx} is empty (length: {episode_length})")
                        return
                    
                    episode_slice = slice(start_idx, end_idx)
                    
                    # Extract episode data
                    tcp_poses = replay_buffer["data"]["TCPPose"][episode_slice]
                    timestamps = replay_buffer["data"]["timestamp"][episode_slice]
                    
                    # Also get grasp data if available
                    grasp_data = None
                    if "Grasp" in replay_buffer["data"]:
                        grasp_data = replay_buffer["data"]["Grasp"][episode_slice]
                    
                    logger.info(f"Replaying episode {episode_idx}: {len(tcp_poses)} steps")
                    logger.info(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
                    
                    # Move to the initial position
                    initial_pose = tcp_poses[0]
                    initial_grasp = 0.0  # Default grasp value if not available
                    if grasp_data is not None and len(grasp_data) > 0:
                        initial_grasp = grasp_data[0]
                    
                    logger.info(f"Moving to initial position: {initial_pose}")
                    env.robot.step(initial_pose, initial_grasp)
                    
                    # Wait 2 seconds
                    logger.info("Waiting 2 seconds before starting replay")
                    time.sleep(2.0)
                    
                    # Start the replay
                    logger.info(f"Starting replay of episode {episode_idx}")
                    
                    # Calculate time offsets from the first timestamp
                    time_offsets = timestamps - timestamps[0]
                    start_time = time.time()
                    
                    for i in range(len(tcp_poses)):
                        # Wait until it's time to execute this action
                        target_time = start_time + time_offsets[i]
                        now = time.time()
                        if target_time > now:
                            time.sleep(target_time - now)
                        
                        # Get the pose and grasp
                        pose = tcp_poses[i]
                        grasp = 0.0
                        if grasp_data is not None:
                            grasp = grasp_data[i]
                        
                        # Execute the step
                        env.robot.step(pose, grasp)
                    
                    logger.info(f"Replay of episode {episode_idx} completed")
                    
                    # Update the target pose after replay
                    nonlocal target_pose
                    target_pose = np.array(tcp_poses[-1], dtype=np.float32)
                    
                except Exception as e:
                    logger.error(f"Error during replay: {str(e)}")
                    traceback.print_exc()

            try:
                logger.info("Starting main control loop...")
                while not stop:

                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + dt
                    t_sample = t_cycle_end - command_latency

                    obs = env.get_obs()

                    press_events = key_counter.get_press_events()

                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            logger.info("Quit requested...")
                            stop = True
                        elif key_stroke == KeyCode(char="c"):
                            # Make sure we have fresh data before starting an episode
                            env.get_obs()
                            time.sleep(0.1)
                            env.start_episode()
                            is_recording = True
                            logger.info("Recording started!")
                        elif key_stroke == KeyCode(char="s"):
                            env.end_episode()
                            is_recording = False
                            logger.info("Recording stopped.")
                        elif key_stroke == Key.backspace:
                            if click.confirm(
                                "Drop the most recently recorded episode?"
                            ):
                                env.drop_episode()
                                is_recording = False
                                logger.info("Episode dropped.")
                        elif key_stroke == KeyCode(char="h"):
                            # Home the robot when 'h' is pressed
                            move_to_home()
                        elif key_stroke == KeyCode(char="r"):
                            # Replay the most recent episode when 'r' is pressed
                            replay_last_episode()

                    stage_val = key_counter[Key.space]

                    precise_wait(t_sample)

                    sm_state = sm.get_motion_state_transformed()

                    dpos = sm_state[:3]
                    drot = sm_state[3:]
                    grasp = sm.grasp

                    input_magnitude = np.linalg.norm(dpos) + np.linalg.norm(drot)
                    significant_movement = input_magnitude > spacemouse_deadzone * 8.0
                    if significant_movement:
                        dpos *= xarm_config.position_gain
                        drot *= xarm_config.orientation_gain

                        curr_rot = st.Rotation.from_euler(
                            "xyz", target_pose[3:], degrees=True
                        )
                        delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                        final_rot = delta_rot * curr_rot

                        target_pose[:3] += dpos
                        target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                        # Grasp does not work.
                        action = np.concatenate([target_pose, [grasp]])

                        exec_timestamp = (
                            t_command_target - time.monotonic() + time.time()
                        )
                        env.exec_actions(
                            actions=[action],
                            timestamps=[exec_timestamp],
                            stages=[stage_val],
                        )
                        logger.debug("Significant movement detected, executing action.")
                    else:
                        action = np.concatenate([target_pose, [grasp]])
                        exec_timestamp = (
                            t_command_target - time.monotonic() + time.time()
                        )
                        env.exec_actions(
                            actions=[action],
                            timestamps=[exec_timestamp],
                            stages=[stage_val],
                        )
                        logger.debug("No significant movement detected.")

                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user.")
            except Exception:
                logger.error("Exception occurred during the main loop:")
                traceback.print_exc()
            finally:
                logger.info("Exiting main loop. Cleaning up...")


if __name__ == "__main__":
    main()
