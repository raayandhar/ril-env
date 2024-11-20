import time
import zarr
import numpy as np
import os
import argparse
import signal

from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig

spacemouse_cfg = SpaceMouseConfig()
xarm_cfg = XArmConfig()
OVERLAY_ALPHA = 0.5

xarm_env = XArmEnv(xarm_cfg)
spacemouse = SpaceMouse(spacemouse_cfg)


def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Stopping xArm recording.")
    global stop_recording
    stop_recording = True


def record_xarm_session(filename):
    global stop_recording
    stop_recording = False

    xarm_env._arm_reset()
    print("Recording xArm... Press Ctrl+C to stop.")

    dpos_record = []
    drot_record = []
    grasp_record = []
    timestamp_record = []

    try:
        while not stop_recording:
            loop_start_time = time.time()

            controller_state = spacemouse.get_controller_state()
            dpos = controller_state["dpos"] * xarm_cfg.position_gain
            drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
            grasp = controller_state["grasp"]

            xarm_env.step(dpos, drot, grasp)

            dpos_record.append(dpos.copy())
            drot_record.append(drot.copy())
            grasp_record.append(grasp)
            timestamp_record.append(loop_start_time)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, xarm_env.control_loop_period - elapsed_time)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"\nAn error occurred during xArm recording: {e}")

    finally:
        save_xarm_recording(
            dpos_record,
            drot_record,
            grasp_record,
            timestamp_record,
            filename,
        )
        xarm_env._arm_reset()
        print("xArm recording session ended.")


def save_xarm_recording(
    dpos_record,
    drot_record,
    grasp_record,
    timestamp_record,
    filename,
):
    recordings_dir = (
        "/home/uril/Github/ril-env/recordings"  # this is manually set to debug
    )
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    filename = os.path.join(recordings_dir, filename)

    if not filename.endswith(".zarr"):
        filename += ".zarr"

    print(f"xArm recording will be saved to '{filename}'.")

    dpos_record = np.array(dpos_record, dtype=np.float32)
    drot_record = np.array(drot_record, dtype=np.float32)
    grasp_record = np.array(grasp_record, dtype=np.float32)
    timestamp_record = np.array(timestamp_record, dtype=np.float64)

    zarr_store = zarr.open(filename, mode="w")
    zarr_store.create_dataset("dpos", data=dpos_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("drot", data=drot_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("grasp", data=grasp_record, chunks=(1,), dtype="float32")
    zarr_store.create_dataset(
        "timestamps", data=timestamp_record, chunks=(1,), dtype="float64"
    )

    print(f"xArm recording saved to '{filename}'.")


def main():
    parser = argparse.ArgumentParser(description="xArm Recording Script")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Filename to save the xArm recording",
    )
    args = parser.parse_args()

    # useful way to handle the script

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    record_xarm_session(args.filename)


if __name__ == "__main__":
    main()
