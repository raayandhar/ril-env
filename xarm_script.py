import time
import zarr
import numpy as np
import sys
import os

from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig

# Configuration
spacemouse_cfg = SpaceMouseConfig()
xarm_cfg = XArmConfig()
shared_directory = "/shared/recordings/"  # Update this path to your shared directory

# Initialize xArm environment and SpaceMouse
xarm_env = XArmEnv(xarm_cfg)
spacemouse = SpaceMouse(spacemouse_cfg)


def record_xarm_session():
    xarm_env._arm_reset()
    print("Recording xArm... Press Ctrl+C to stop.")

    filename = input("Enter filename to save the xArm recording (save as .zarr): ")
    filename = os.path.join(shared_directory, filename)

    dpos_record = []
    drot_record = []
    grasp_record = []
    timestamp_record = []

    try:
        while True:
            loop_start_time = time.time()

            # Get controller state
            controller_state = spacemouse.get_controller_state()
            dpos = controller_state["dpos"] * xarm_cfg.position_gain
            drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
            grasp = controller_state["grasp"]

            # Step the xArm
            xarm_env.step(dpos, drot, grasp)

            # Record data
            dpos_record.append(dpos.copy())
            drot_record.append(drot.copy())
            grasp_record.append(grasp)
            timestamp_record.append(loop_start_time)

            # Maintain control loop period
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0.0, xarm_env.control_loop_period - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nxArm recording stopped by user.")

    # Save the recording
    save_xarm_recording(
        dpos_record,
        drot_record,
        grasp_record,
        timestamp_record,
        filename,
    )
    xarm_env._arm_reset()


def save_xarm_recording(
    dpos_record,
    drot_record,
    grasp_record,
    timestamp_record,
    filename,
):
    if not filename.endswith(".zarr"):
        filename += ".zarr"

    # Convert lists to numpy arrays
    dpos_record = np.array(dpos_record, dtype=np.float32)
    drot_record = np.array(drot_record, dtype=np.float32)
    grasp_record = np.array(grasp_record, dtype=np.float32)
    timestamp_record = np.array(timestamp_record, dtype=np.float64)

    # Save data to Zarr format
    zarr_store = zarr.open(filename, mode="w")
    zarr_store.create_dataset("dpos", data=dpos_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("drot", data=drot_record, chunks=(1, 3), dtype="float32")
    zarr_store.create_dataset("grasp", data=grasp_record, chunks=(1,), dtype="float32")
    zarr_store.create_dataset(
        "timestamps", data=timestamp_record, chunks=(1,), dtype="float64"
    )

    print(f"xArm recording saved to '{filename}'.")


def main():
    record_xarm_session()


if __name__ == "__main__":
    main()
