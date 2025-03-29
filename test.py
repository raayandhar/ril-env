# test_multiprocessing.py
from ril_env.realsense import SingleRealsense
from multiprocessing.managers import SharedMemoryManager

if __name__ == "__main__":
    with SharedMemoryManager() as shm:
        cam = SingleRealsense(shm_manager=shm, serial_number="317222072157", verbose=True)
        cam.start(wait=True)
        print("Camera is ready!")
        cam.stop(wait=True)
        print("Stopped camera.")
