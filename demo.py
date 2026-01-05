# import cv2

# def find_external_camera(max_index=5):
#     working = []

#     for i in range(max_index):
#         cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 working.append(i)
#             cap.release()

#     return working
# print("Searching for external cameras...")
# cameras = find_external_camera()
# print(f"Available camera indices: {cameras}")   


import cv2
import numpy as np
import pyrealsense2 as rs
import time

# =========================
# CONFIG
# =========================
WIDTH, HEIGHT = 1280, 720
RGB_CAM_INDEX = 2   # external USB camera
SAVE_KEY = ord('s')
EXIT_KEY = 27

# =========================
# STEP 1: CHECK REALSENSE
# =========================
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    raise RuntimeError("❌ Intel RealSense NOT connected")

print("✅ RealSense detected")

# =========================
# STEP 2: START REALSENSE
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm-up frames (IMPORTANT)
for _ in range(50):
    pipeline.wait_for_frames()

profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("✅ RealSense pipeline started")

# =========================
# STEP 3: OPEN RGB CAMERA
# =========================
rgb_cap = cv2.VideoCapture(RGB_CAM_INDEX, cv2.CAP_DSHOW)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not rgb_cap.isOpened():
    pipeline.stop()
    raise RuntimeError("❌ External RGB camera NOT accessible")

print("✅ External RGB camera opened")

# =========================
# MAIN LOOP (EXAMPLE)
# =========================
while True:

    ret, rgb_frame = rgb_cap.read()
    if not ret:
        print("RGB frame grab failed")
        break

    cv2.imshow("External RGB Live", rgb_frame)

    frames = pipeline.poll_for_frames()
    if frames:
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        if depth:
            pass  # depth available

    key = cv2.waitKey(1) & 0xFF
    if key == EXIT_KEY:
        break

# =========================
# CLEANUP
# =========================
rgb_cap.release()
pipeline.stop()
cv2.destroyAllWindows()
