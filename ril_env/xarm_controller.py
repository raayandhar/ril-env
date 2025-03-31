import os
import time
import enum
import multiprocessing as mp
import numpy as np
import logging
from multiprocessing.managers import SharedMemoryManager

from xarm.wrapper import XArmAPI
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

# Setup logging similar to the RTDE example
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Command enum matching RTDE's style
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2  # Kept for compatibility, even if not used

class XArmController(mp.Process):
    """
    XArmController replicates the structure of RTDEInterpolationController:
      - Uses shared memory queues to receive commands.
      - Uses a ring buffer to store state, seeded with an example that includes a relative robot timestamp.
      - In its control loop, it processes commands and directly commands the robot via set_servo_cartesian.
    """
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 ip: str = "192.168.1.223",
                 frequency: int = 50,
                 soft_real_time: bool = False,
                 verbose: bool = True):
        super().__init__(name="XArmController")
        self.ip = ip
        self.frequency = frequency
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Events for synchronization
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        # --- Build Input Queue ---
        # The example command has the same keys as the RTDE controller.
        queue_example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros(6, dtype=np.float64),  # [x, y, z, roll, pitch, yaw]
            'duration': 0.0,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=queue_example,
            buffer_size=256
        )

        # --- Build Ring Buffer ---
        # To mimic the RTDE approach, we first connect temporarily to the xArm to obtain initial state.
        try:
            arm_temp = XArmAPI(self.ip)
            arm_temp.connect()
            arm_temp.clean_error()
            arm_temp.clean_warn()
            code = arm_temp.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"motion_enable error: {code}")
            code = arm_temp.set_mode(1)
            if code != 0:
                raise RuntimeError(f"set_mode error: {code}")
            code = arm_temp.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state error: {code}")

            # Mimic the RTDE 'receive_keys' approach using a list of state keys.
            receive_keys = ['TCPPose', 'TCPSpeed', 'JointAngles', 'JointSpeeds']
            state_example = {}

            # Get TCPPose: use get_position (first 6 values: x,y,z,roll,pitch,yaw)
            code, pos = arm_temp.get_position(is_radian=False)
            if code == 0:
                state_example['TCPPose'] = np.array(pos[:6], dtype=np.float64)
            else:
                state_example['TCPPose'] = np.zeros(6, dtype=np.float64)

            # Get TCPSpeed: handle callable or value directly.
            try:
                if callable(arm_temp.realtime_tcp_speed):
                    tcp_speed = arm_temp.realtime_tcp_speed()
                else:
                    tcp_speed = arm_temp.realtime_tcp_speed
                state_example['TCPSpeed'] = np.array(tcp_speed, dtype=np.float64)
            except Exception:
                state_example['TCPSpeed'] = np.zeros(6, dtype=np.float64)

            # Get JointAngles: use get_servo_angle()
            code, angles = arm_temp.get_servo_angle(is_radian=False)
            if code == 0:
                state_example['JointAngles'] = np.array(angles, dtype=np.float64)
            else:
                state_example['JointAngles'] = np.zeros(7, dtype=np.float64)

            # Get JointSpeeds: handle callable or value directly.
            try:
                if callable(arm_temp.realtime_joint_speeds):
                    joint_speeds = arm_temp.realtime_joint_speeds()
                else:
                    joint_speeds = arm_temp.realtime_joint_speeds
                state_example['JointSpeeds'] = np.array(joint_speeds, dtype=np.float64)
            except Exception:
                state_example['JointSpeeds'] = np.zeros(7, dtype=np.float64)

            # Include a robot timestamp similar to the RTDE controller (absolute for now).
            state_example['robot_receive_timestamp'] = time.time()

            self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=state_example,
                get_max_k=128,
                get_time_budget=0.2,
                put_desired_frequency=frequency
            )

            # Disconnect the temporary connection; the main loop will reconnect.
            arm_temp.disconnect()

        except Exception as e:
            logger.error(f"Error during initial state fetch: {e}")
            raise e

        # Store the last target pose; initialize it from the example.
        self.last_target_pose = state_example['TCPPose']

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(3)
            assert self.is_alive(), "XArmController did not start correctly."
        if self.verbose:
            logger.debug(f"[XArmController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        # Place a STOP command into the input queue and set the stop event.
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        self.stop_event.set()
        if wait:
            self.join()

    def get_state(self, k=None):
        if k is None:
            return self.ring_buffer.get()
        else:
            return self.ring_buffer.get_last_k(k)

    def run(self):
        """
        Main loop:
          - Reconnects to xArm and initializes it.
          - Enters a loop running at the desired frequency.
          - Processes all commands from the shared memory queue.
          - Commands the robot using set_servo_cartesian with the latest target pose.
          - Fetches robot state using API calls (mimicking getattr style) and stores it in the ring buffer.
          - The 'robot_receive_timestamp' is now stored as relative time since this process started.
        """
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                logger.error(f"Real-time scheduling failed: {e}")

        try:
            logger.info(f"[XArmController] Connecting to xArm at {self.ip}")
            arm = XArmAPI(self.ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()

            code = arm.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"motion_enable error: {code}")
            code = arm.set_mode(1)
            if code != 0:
                raise RuntimeError(f"set_mode error: {code}")
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state error: {code}")

            # Initialize last_target_pose using the current robot pose.
            code, pos = arm.get_position(is_radian=False)
            if code == 0:
                self.last_target_pose = np.array(pos[:6], dtype=np.float64)
            else:
                logger.error("Failed to get initial position; defaulting to zeros.")
                self.last_target_pose = np.zeros(6, dtype=np.float64)

            # Record the start time for relative timestamping.
            start_time = time.time()

            # Signal that the controller is ready.
            self.ready_event.set()

            dt = 1.0 / self.frequency
            iter_idx = 0

            while not self.stop_event.is_set():
                t_start = time.time()

                # Process all commands from the input queue.
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: commands[key][i] for key in commands}
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        logger.debug("[XArmController] Received STOP command.")
                        self.stop_event.set()
                        break
                    elif cmd == Command.SERVOL.value:
                        # Directly update the target pose.
                        target_pose = np.array(command['target_pose'], dtype=np.float64)
                        self.last_target_pose = target_pose
                        if self.verbose:
                            logger.debug(f"[XArmController] New target pose: {target_pose}")
                    else:
                        logger.error(f"[XArmController] Unknown command: {cmd}")

                # Command the robot via set_servo_cartesian using the latest target pose.
                code = arm.set_servo_cartesian(list(self.last_target_pose), is_radian=False)
                if code != 0:
                    logger.error(f"[XArmController] set_servo_cartesian error: {code}")

                # --- Update Robot State ---
                state = {}
                # Fetch TCPPose using get_position
                code, pos = arm.get_position(is_radian=False)
                if code == 0:
                    state['TCPPose'] = np.array(pos[:6], dtype=np.float64)
                else:
                    state['TCPPose'] = np.zeros(6, dtype=np.float64)

                # Fetch TCPSpeed, handling callable or attribute
                try:
                    if callable(arm.realtime_tcp_speed):
                        tcp_speed = arm.realtime_tcp_speed()
                    else:
                        tcp_speed = arm.realtime_tcp_speed
                    state['TCPSpeed'] = np.array(tcp_speed, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_tcp_speed: {e}")
                    state['TCPSpeed'] = np.zeros(6, dtype=np.float64)

                # Fetch JointAngles using get_servo_angle
                code, angles = arm.get_servo_angle(is_radian=False)
                if code == 0:
                    state['JointAngles'] = np.array(angles, dtype=np.float64)
                else:
                    state['JointAngles'] = np.zeros(7, dtype=np.float64)

                # Fetch JointSpeeds, handling callable or attribute
                try:
                    if callable(arm.realtime_joint_speeds):
                        joint_speeds = arm.realtime_joint_speeds()
                    else:
                        joint_speeds = arm.realtime_joint_speeds
                    state['JointSpeeds'] = np.array(joint_speeds, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_joint_speeds: {e}")
                    state['JointSpeeds'] = np.zeros(7, dtype=np.float64)

                # Add a robot timestamp as elapsed time since start.
                state['robot_receive_timestamp'] = time.time() - start_time

                # Store the current state into the ring buffer
                self.ring_buffer.put(state)

                # Regulate loop frequency.
                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                iter_idx += 1
                if self.verbose:
                    logger.debug(f"[XArmController] Iteration {iter_idx} at {1.0/(time.time()-t_start):.2f} Hz")
        except Exception as e:
            logger.error(f"[XArmController] Exception in control loop: {e}")
        finally:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                logger.info(f"[XArmController] Disconnected from xArm at {self.ip}")
            except Exception as e:
                logger.error(f"[XArmController] Cleanup error: {e}")
            self.ready_event.set()
