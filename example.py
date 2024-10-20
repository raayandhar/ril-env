import time
import cv2
from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig
from ril_env.camera.rs_streamer import RealsenseStreamer
from threading import Thread

spacemouse_cfg = SpaceMouseConfig()
xarm_cfg = XArmConfig()

xarm_env = XArmEnv(xarm_cfg)
spacemouse = SpaceMouse(spacemouse_cfg)

external_streamer = RealsenseStreamer("317422075456")
internal_streamer = RealsenseStreamer("317222072157")

control_loop_rate = xarm_cfg.control_loop_rate
control_loop_period = 1.0 / control_loop_rate


def capture_images():
    global ext_rgb_img, int_rgb_img
    while True:
        _, ext_rgb_img, _, _ = external_streamer.capture_rgbd()
        _, int_rgb_img, _, _ = internal_streamer.capture_rgbd()


image_thread = Thread(target=capture_images)
image_thread.daemon = True
image_thread.start()


xarm_env._arm_reset()

try:
    while True:
        loop_start_time = time.time()
        controller_state = spacemouse.get_controller_state()
        dpos = controller_state["dpos"] * xarm_cfg.position_gain
        drot = controller_state["raw_drotation"] * xarm_cfg.orientation_gain
        grasp = controller_state["grasp"]

        xarm_env.step(dpos, drot, grasp)

        cv2.waitKey(1)
        cv2.imshow("ext", ext_rgb_img)
        cv2.imshow("int", int_rgb_img)

        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0.0, control_loop_period - elapsed_time)
        time.sleep(sleep_time)
except KeyboardInterrupt:
    print("Teleop manually shut down!")
    xarm_env._arm_reset()
except Exception as e:
    print(f"An error occurred: {e}")
    xarm_env._arm_reset()
