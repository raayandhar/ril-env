#!/bin/bash

# This is all hardcoded!!

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 xarm_filename camera_filename"
    exit 1
fi

XARM_RECORDING_FILENAME="$1"
CAMERA_RECORDING_FILENAME="$2"

if [[ "$XARM_RECORDING_FILENAME" != *.zarr ]]; then
    XARM_RECORDING_FILENAME="${XARM_RECORDING_FILENAME}.zarr"
fi
if [[ "$CAMERA_RECORDING_FILENAME" != *.zarr ]]; then
    CAMERA_RECORDING_FILENAME="${CAMERA_RECORDING_FILENAME}.zarr"
fi

NUC_USER="uril"
NUC_HOST="192.168.1.3"
NUC_SCRIPT_PATH="/home/uril/Github/ril-env/xarm_script.py"
CAMERA_SCRIPT_PATH="/home/u-ril/Github/ril-env/camera_script.py"

NUC_RECORDING_DIR="/home/uril/Github/ril-env/recordings"
NUC_RECORDING_PATH="$NUC_RECORDING_DIR/$XARM_RECORDING_FILENAME"

LOCAL_RECORDING_DIR="/home/u-ril/Github/ril-env/recordings"
LOCAL_RECORDING_PATH="$LOCAL_RECORDING_DIR/$XARM_RECORDING_FILENAME"

function cleanup {
    echo "Caught SIGINT signal! Terminating processes..."

    trap - SIGINT

    if kill -0 $CAMERA_PID 2>/dev/null; then
        kill -SIGINT $CAMERA_PID
    fi

    ssh $NUC_USER@$NUC_HOST "pkill -SIGINT -f $NUC_SCRIPT_PATH"

    wait $CAMERA_PID $SSH_PID

    echo "Processes terminated. Exiting script."
}

trap cleanup SIGINT

ssh -tt $NUC_USER@$NUC_HOST "bash -ic 'conda activate xarm && python3 $NUC_SCRIPT_PATH --filename \"$XARM_RECORDING_FILENAME\"'" &
SSH_PID=$!

source /home/u-ril/miniconda3/etc/profile.d/conda.sh
conda activate xarm

python3 $CAMERA_SCRIPT_PATH --filename "$CAMERA_RECORDING_FILENAME" &
CAMERA_PID=$!

echo "Starting recordings on both devices..."

wait $SSH_PID $CAMERA_PID

scp -r $NUC_USER@$NUC_HOST:"$NUC_RECORDING_PATH" "$LOCAL_RECORDING_DIR/"

echo "Recording session completed and data transferred."
