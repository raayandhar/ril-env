import time
import scipy.spatial.transform as st

from multiprocessing.managers import SharedMemoryManager
from ril_env.spacemouse import Spacemouse
from ril_env.xarm_controller import XArmController, Command

def main():
    with SharedMemoryManager() as shm_manager, Spacemouse(deadzone=0.4) as sm:
        xarm_ctrl = XArmController(
            shm_manager=shm_manager,
            ip="192.168.1.223",
            frequency=50,
            verbose=True
        )
        xarm_ctrl.start(wait=True)
        print("XArmController started and ready.")

        # Use the controller's last known target pose as the starting point.
        current_target_pose = xarm_ctrl.last_target_pose.copy()

        last_timestamp = None
        try:
            while True:
                loop_start = time.monotonic()
                
                # Get spacemouse 6D motion.
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                # Replicating .step() here
                dpos *= xarm_ctrl.position_gain
                drot *= xarm_ctrl.orientation_gain

                curr_rot = st.Rotation.from_euler("xyz", xarm_ctrl.current_orientation, degrees=True)
                delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                final_rot = delta_rot * curr_rot

                current_target_pose[:3] += dpos
                current_target_pose[3:] = final_rot.as_euler("xyz", degrees=True)
                
                # Update target pose (with scaling if needed).
                # current_target_pose[:3] += dpos
                # current_target_pose[3:] += drot
                # Need to test this
                # Create and send a STEP command.
                command = {
                    'cmd': Command.STEP.value,
                    'target_pose': current_target_pose,
                    'grasp': grasp,
                    'duration': 0.02,
                    'target_time': time.time() + 0.02
                }
                """
                command = {
                    'cmd': Command.STEP.value,
                    'dpos': dpos,
                    'drot': drot,
                    'grasp': grasp,
                    'duration': 0.02,
                    'target_time': time.time() + 0.02,
                }
                """
                xarm_ctrl.input_queue.put(command)

                # Fetch the latest state from the ring buffer.
                state = xarm_ctrl.get_state()
                ts = state.get('robot_receive_timestamp')
                # Check if the timestamp has updated.
                if ts != last_timestamp:
                    print(f"Ring buffer updated, timestamp: {ts:.3f}")
                    last_timestamp = ts
                # TODO: Check and print the data we are receiving
                # Maintain the control loop frequency (50 Hz).
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0, 0.02 - elapsed))
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            xarm_ctrl.stop(wait=True)
            print("XArmController stopped.")

if __name__ == "__main__":
    main()
