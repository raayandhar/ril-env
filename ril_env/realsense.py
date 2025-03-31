import enum
import time
import json
import numpy as np
import multiprocessing as mp
import pyrealsense2 as rs

from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Callable, Dict
from ril_env.timestamp_accumulator import get_accumulate_timestamp_idxs
from ril_env.video_recorder import VideoRecorder
from shared_memory.shared_ndarray import SharedNDArray
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number: int,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config=None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples["infrared"] = np.empty(shape=shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=(
                examples if vis_transform is None else vis_transform(dict(examples))
            ),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=18,
                thread_type="FRAME",
                thread_count=1,
            )

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == "D400":
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        print("start")
        super().start()
        print("super() start finished")
        print("wait: ", wait)
        if wait:
            print("starting the wait")
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        print("start_wait")
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put(
            {
                "cmd": Command.SET_COLOR_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale

    def start_recording(self, video_path: str, start_time: float = -1):
        assert self.enable_color

        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    # ========= interval API ===========
    def run(self):
        print("HELLO!")
        # limit threads
        threadpool_limits(1)
        print("HELLO?")
        #cv2.setNumThreads(1)
        print("HELLO2")
        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)
        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            print("Trying")
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            print("Passing")

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ["fx", "fy", "ppx", "ppy", "height", "width"]
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale

            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                print("wait for frames to come in")
                frameset = pipeline.wait_for_frames()
                print("we good!")
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                # grab data
                data = dict()
                data["camera_receive_timestamp"] = receive_time
                # realsense report in ms
                data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data["color"] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data["camera_capture_timestamp"] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    data["depth"] = np.asarray(frameset.get_depth_frame().get_data())
                if self.enable_infrared:
                    data["infrared"] = np.asarray(
                        frameset.get_infrared_frame().get_data()
                    )

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True,
                    )

                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data["timestamp"] = receive_time
                        # print(step_idx, data['timestamp'])
                        # SWITCHED TO wait=True
                        self.ring_buffer.put(put_data, wait=True)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    # SWITCHED TO wait=True
                    self.ring_buffer.put(put_data, wait=True)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(
                        rec_data["color"], frame_time=receive_time
                    )

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command["option_enum"])
                        value = float(command["option_value"])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command["option_enum"])
                        value = float(command["option_value"])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time = command["recording_start_time"]
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]
                        # self.ring_buffer.clear()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")


if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        # Get connected RealSense devices (only D400 series are considered).
        serials = SingleRealsense.get_connected_devices_serial()
        if not serials:
            print("No RealSense devices found.")
            exit(1)

        # Select the first available camera.
        serial = serials[0]
        print(serials[0])
        print(f"Using camera with serial: {serial}")

        # Create an instance of SingleRealsense.
        # Adjust parameters as needed.
        camera = SingleRealsense(
            shm_manager=shm_manager,
            serial_number=serial,  # note: parameter name in the constructor is 'serial_numer'
            resolution=(1280, 720),
            capture_fps=30,
            verbose=True,
        )

        # Start the camera process and wait until it's ready.
        camera.start(wait=True)
        print("Camera process started and is ready.")

        # Give a moment for the camera to warm up.
        time.sleep(1)

        # Specify the path for the video file.
        video_path = "test_video.mp4"
        print(f"Starting video recording to {video_path}...")

        # Start recording; here we use the current time as the start time.
        camera.start_recording(video_path, start_time=time.time())

        # Record for 5 seconds.
        time.sleep(5)

        # Stop recording.
        camera.stop_recording()
        print("Recording stopped.")

        # Optionally, let the camera process run a little longer to flush frames.
        time.sleep(2)

        # Stop the camera process.
        camera.stop(wait=True)
        print("Camera process terminated.")

"""
single_realsense.py
An ultra-debugging version that ensures we know if child processes even
begin their run() method, or crash on import, or fail for other reasons.

import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
import sys
import traceback

from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Callable, Dict

from ril_env.timestamp_accumulator import get_accumulate_timestamp_idxs
from ril_env.video_recorder import VideoRecorder

from shared_memory.shared_ndarray import SharedNDArray
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty

import logging

logger = logging.getLogger(__name__)

print(
    f"[single_realsense.py top-level] Imported in PID={mp.current_process().pid}. "
    f"__name__={__name__}"
)

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number: str,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config=None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()
        self.daemon = True
        self.serial_number = str(serial_number)

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        shape = resolution[::-1]
        examples = {}
        if enable_color:
            examples["color"] = np.zeros(shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.zeros(shape, dtype=np.uint16)
        if enable_infrared:
            examples["infrared"] = np.zeros(shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        if transform is not None:
            test_out = transform(dict(examples))
            if "color" in test_out and test_out["color"].shape != examples["color"].shape:
                logger.info(
                    f"[SingleRealsense {self.serial_number}] transform changes 'color' shape from "
                    f"{examples['color'].shape} to {test_out['color'].shape}"
                )

        main_examples = examples if transform is None else transform(dict(examples))
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=main_examples,
            get_max_k=get_max_k,
            get_time_budget=0.5,
            put_desired_frequency=put_fps,
        )

        vis_examples = examples if vis_transform is None else vis_transform(dict(examples))
        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples,
            get_max_k=1,
            get_time_budget=0.5,
            put_desired_frequency=capture_fps,
        )

        cmd_examples = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array(" " * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_examples,
            buffer_size=64,
        )

        self.intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        self.intrinsics_array.get()[:] = 0

        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=18,
                thread_type="FRAME",
                thread_count=1,
            )

        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    @staticmethod
    def get_connected_devices_serial():
        ctx = rs.context()
        serials = []
        for device in ctx.devices:
            name = device.get_info(rs.camera_info.name)
            if name.lower() != "platform camera":
                product_line = device.get_info(rs.camera_info.product_line)
                serial = device.get_info(rs.camera_info.serial_number)
                if product_line == "D400":
                    serials.append(serial)
        serials.sort()
        return serials

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        print(
            f"[SingleRealsense {self.serial_number}] (PID={mp.current_process().pid}) before super().start()"
        )
        super().start()
        print(
            f"[SingleRealsense {self.serial_number}] (PID={mp.current_process().pid}) after super().start()"
        )
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    def run(self):
        print(
            f"[CHILD {self.serial_number}] run() entered in PID={mp.current_process().pid}!"
        )
        try:
            self._run_impl()
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(
                f"[CHILD {self.serial_number}] CRASHED with {e}\n"
            )
            # We still set ready_event so main doesn't hang
            self.ready_event.set()
        finally:
            print(f"[CHILD {self.serial_number}] run() FINALLY block in PID={mp.current_process().pid}.")

    def _run_impl(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)

        logger.info(
            f"[SingleRealsense {self.serial_number}] (child PID={mp.current_process().pid}) Starting pipeline..."
        )

        pipeline = rs.pipeline()
        cfg = rs.config()
        w, h = self.resolution
        fps = self.capture_fps

        if self.enable_color:
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        if self.enable_infrared:
            cfg.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)

        cfg.enable_device(self.serial_number)
        print(f"[CHILD {self.serial_number}] about to do pipeline.start() in PID={os.getpid()}")
        pipeline_profile = pipeline.start(cfg)
        print(f"[CHILD {self.serial_number}] pipeline.start() returned in PID={os.getpid()}")
        logger.info(
            f"[SingleRealsense {self.serial_number}] Pipeline started OK (child PID={mp.current_process().pid})."
        )

        try:
            dev = pipeline_profile.get_device()
            color_sensor = dev.first_color_sensor()
            color_sensor.set_option(rs.option.global_time_enabled, 1)
        except Exception as ex:
            logger.warning(
                f"[SingleRealsense {self.serial_number}] Could not enable global time: {ex}"
            )

        if self.advanced_mode_config is not None:
            try:
                adv_dev = rs.rs400_advanced_mode(dev)
                adv_dev.load_json(json.dumps(self.advanced_mode_config))
                logger.info(
                    f"[SingleRealsense {self.serial_number}] Loaded advanced mode config."
                )
            except Exception as ex:
                logger.warning(
                    f"[SingleRealsense {self.serial_number}] Failed to load advanced config: {ex}"
                )

        color_stream = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        for i, name in enumerate(["fx", "fy", "ppx", "ppy", "height", "width"]):
            self.intrinsics_array.get()[i] = getattr(intr, name)

        if self.enable_depth:
            depth_scale = dev.first_depth_sensor().get_depth_scale()
            self.intrinsics_array.get()[-1] = depth_scale

        align_to_color = rs.align(rs.stream.color)
        frames_received = False

        if self.put_start_time is None:
            self.put_start_time = time.time()
        put_idx = None

        logger.info(
            f"[SingleRealsense {self.serial_number}] Attempting to get frames (child PID={mp.current_process().pid})."
        )

        while not self.stop_event.is_set():
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as ex:
                logger.error(
                    f"[SingleRealsense {self.serial_number}] wait_for_frames timed out after 5s! {ex}"
                )
                break

            receive_time = time.time()
            frameset = align_to_color.process(frameset)

            data = {}
            data["camera_receive_timestamp"] = receive_time
            first_frame = frameset.get_color_frame() if self.enable_color else frameset.get_depth_frame()
            if first_frame:
                data["camera_capture_timestamp"] = first_frame.get_timestamp() / 1000.0
            else:
                data["camera_capture_timestamp"] = receive_time

            if self.enable_color:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    data["color"] = np.asanyarray(color_frame.get_data())

            if self.enable_depth:
                depth_frame = frameset.get_depth_frame()
                if depth_frame:
                    data["depth"] = np.asanyarray(depth_frame.get_data())

            if self.enable_infrared:
                ir_frame = frameset.get_infrared_frame()
                if ir_frame:
                    data["infrared"] = np.asanyarray(ir_frame.get_data())

            if self.transform:
                put_data = self.transform(dict(data))
            else:
                put_data = data

            if self.put_downsample:
                local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                    timestamps=[receive_time],
                    start_time=self.put_start_time,
                    dt=1.0 / self.put_fps,
                    next_global_idx=put_idx,
                    allow_negative=True,
                )
                for step_idx in global_idxs:
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    self.ring_buffer.put(put_data, wait=True)
            else:
                step_idx = int((receive_time - self.put_start_time) * self.put_fps)
                put_data["step_idx"] = step_idx
                put_data["timestamp"] = receive_time
                self.ring_buffer.put(put_data, wait=True)

            # vis ring buffer
            if self.vis_transform == self.transform:
                vis_data = put_data
            elif self.vis_transform:
                vis_data = self.vis_transform(dict(data))
            else:
                vis_data = data
            self.vis_ring_buffer.put(vis_data, wait=False)

            # video recording
            rec_data = data
            if self.recording_transform == self.transform:
                rec_data = put_data
            elif self.recording_transform:
                rec_data = self.recording_transform(dict(data))

            if self.video_recorder.is_ready():
                if "color" in rec_data:
                    self.video_recorder.write_frame(
                        rec_data["color"], frame_time=receive_time
                    )

            if not frames_received:
                frames_received = True
                self.ready_event.set()
                logger.info(
                    f"[SingleRealsense {self.serial_number}] Got frameset! Marking camera as ready (child PID={mp.current_process().pid})."
                )

            if self.verbose:
                now = time.time()
                dt = now - getattr(self, "_last_time", now)
                setattr(self, "_last_time", now)
                if dt > 0:
                    fps_approx = 1.0 / dt
                    logger.debug(
                        f"[SingleRealsense {self.serial_number}] Stream FPS ~ {fps_approx:.1f}"
                    )

            try:
                commands = self.command_queue.get_all()
                n_cmd = len(commands["cmd"])
            except Empty:
                n_cmd = 0

            for i in range(n_cmd):
                cmdval = commands["cmd"][i]
                if cmdval == Command.SET_COLOR_OPTION.value:
                    sensor = pipeline_profile.get_device().first_color_sensor()
                    option = rs.option(commands["option_enum"][i])
                    val = float(commands["option_value"][i])
                    sensor.set_option(option, val)
                    logger.info(
                        f"[SingleRealsense {self.serial_number}] Set color option {option} = {val}"
                    )
                elif cmdval == Command.SET_DEPTH_OPTION.value:
                    sensor = pipeline_profile.get_device().first_depth_sensor()
                    option = rs.option(commands["option_enum"][i])
                    val = float(commands["option_value"][i])
                    sensor.set_option(option, val)
                    logger.info(
                        f"[SingleRealsense {self.serial_number}] Set depth option {option} = {val}"
                    )
                elif cmdval == Command.START_RECORDING.value:
                    path = (
                        commands["video_path"][i]
                        .tobytes()
                        .decode("utf-8")
                        .rstrip("\x00")
                    )
                    rec_start_time = commands["recording_start_time"][i]
                    if rec_start_time < 0:
                        rec_start_time = None
                    logger.info(
                        f"[SingleRealsense {self.serial_number}] START_RECORDING => {path} start_time={rec_start_time}"
                    )
                    self.video_recorder.start(path, start_time=rec_start_time)
                elif cmdval == Command.STOP_RECORDING.value:
                    logger.info(
                        f"[SingleRealsense {self.serial_number}] STOP_RECORDING => flush & reset put_idx"
                    )
                    self.video_recorder.stop()
                    put_idx = None
                elif cmdval == Command.RESTART_PUT.value:
                    put_idx = None
                    new_start = commands["put_start_time"][i]
                    logger.info(
                        f"[SingleRealsense {self.serial_number}] RESTART_PUT => put_start_time={new_start}"
                    )
                    self.put_start_time = new_start

        logger.info(
            f"[SingleRealsense {self.serial_number}] Shutting down pipeline (child PID={mp.current_process().pid})."
        )
        self.video_recorder.stop()
        try:
            pipeline.stop()
        except:
            pass

        self.ready_event.set()
        logger.info(
            f"[SingleRealsense {self.serial_number}] Exiting worker process gracefully (child PID={mp.current_process().pid})."
        )

    #
    # Public API
    #
    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def set_color_option(self, option: rs.option, value: float):
        cmd = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": option.value,
            "option_value": float(value),
        }
        self.command_queue.put(cmd)

    def set_exposure(self, exposure=None, gain=None):
        if exposure is None and gain is None:
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, float(white_balance))

    def get_intrinsics(self):
        return self.intrinsics_array.get()[:4].copy()

    def get_depth_scale(self):
        return self.intrinsics_array.get()[-1]

    def start_recording(self, video_path: str, start_time: float = -1):
        if len(video_path.encode("utf-8")) > self.MAX_PATH_LENGTH:
            raise RuntimeError("Video path too long!")
        cmd = {
            "cmd": Command.START_RECORDING.value,
            "video_path": np.frombuffer(video_path.encode("utf-8"), dtype=np.uint8),
            "recording_start_time": float(start_time),
        }
        self.command_queue.put(cmd)

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time: float):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": float(start_time)}
        )
"""
