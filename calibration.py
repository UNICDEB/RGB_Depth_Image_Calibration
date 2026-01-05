import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ==============================
# CONFIG
# ==============================
WIDTH, HEIGHT = 1280, 720
RGB_CAM_INDEX = 2        # <-- CHANGE if needed
SAVE_KEY = ord('s')
EXIT_KEY = 27

# ==============================
# OPEN EXTERNAL RGB CAMERA FIRST
# ==============================
rgb_cap = cv2.VideoCapture(RGB_CAM_INDEX, cv2.CAP_DSHOW)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not rgb_cap.isOpened():
    raise RuntimeError("External RGB camera NOT accessible")

print("External RGB camera locked")

# ==============================
# INIT REALSENSE SAFELY
# ==============================
ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise RuntimeError("No RealSense detected")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm-up RealSense (IMPORTANT)
for _ in range(60):
    pipeline.wait_for_frames()

depth_scale = pipeline.get_active_profile() \
    .get_device().first_depth_sensor().get_depth_scale()

print("RealSense started correctly")

# ==============================
# FEATURE MATCHING
# ==============================
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("===================================")
print("LIVE: External RGB camera")
print("Press 'S' → Save COMMON FOV")
print("Press ESC → Exit")
print("===================================")

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, rgb_frame = rgb_cap.read()
    if not ret:
        print("RGB frame grab failed")
        break

    rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
    cv2.imshow("External RGB Live Feed", rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == SAVE_KEY:
        print("Saving common FOV...")

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("RealSense frame error")
            continue

        rs_color = np.asanyarray(color_frame.get_data())
        rs_depth = np.asanyarray(depth_frame.get_data())

        # ---------- FEATURE MATCH ----------
        gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        gray_rs = cv2.cvtColor(rs_color, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(gray_rgb, None)
        kp2, des2 = orb.detectAndCompute(gray_rs, None)

        if des1 is None or des2 is None:
            print("No common area")
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 30:
            print("No common area")
            continue

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
        if H is None:
            print("Homography failed")
            continue

        # ---------- WARP ----------
        rs_color_warped = cv2.warpPerspective(rs_color, H, (WIDTH, HEIGHT))
        rs_depth_warped = cv2.warpPerspective(rs_depth, H, (WIDTH, HEIGHT))

        mask = np.any(rs_color_warped > 0, axis=2)
        if np.count_nonzero(mask) < 5000:
            print("No common area")
            continue

        ys, xs = np.where(mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        rgb_common = rgb_frame[y1:y2, x1:x2]
        rs_color_common = rs_color_warped[y1:y2, x1:x2]
        rs_depth_common = rs_depth_warped[y1:y2, x1:x2]

        # ---------- SAVE ----------
        cv2.imwrite("rgb_common.png", rgb_common)
        cv2.imwrite("rs_color_common.png", rs_color_common)

        with open("depth_common.txt", "w") as f:
            h, w = rs_depth_common.shape
            for y in range(h):
                for x in range(w):
                    d = rs_depth_common[y, x] * depth_scale
                    if d > 0:
                        f.write(f"{x},{y},{d:.4f}\n")

        print("Common FOV SAVED successfully")

    elif key == EXIT_KEY:
        break

# ==============================
# CLEANUP
# ==============================
rgb_cap.release()
pipeline.stop()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import pyrealsense2 as rs
# import time

# # =========================
# # CONFIG
# # =========================
# WIDTH, HEIGHT = 1280, 720
# RGB_CAM_INDEX = 2   # external USB camera
# SAVE_KEY = ord('s')
# EXIT_KEY = 27

# # =========================
# # STEP 1: CHECK REALSENSE
# # =========================
# ctx = rs.context()
# devices = ctx.query_devices()

# if len(devices) == 0:
#     raise RuntimeError("❌ Intel RealSense NOT connected")

# print("✅ RealSense detected")

# # =========================
# # STEP 2: START REALSENSE
# # =========================
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

# pipeline.start(config)
# align = rs.align(rs.stream.color)

# # Warm-up frames (IMPORTANT)
# for _ in range(50):
#     pipeline.wait_for_frames()

# profile = pipeline.get_active_profile()
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()

# print("✅ RealSense pipeline started")

# # =========================
# # STEP 3: OPEN RGB CAMERA
# # =========================
# rgb_cap = cv2.VideoCapture(RGB_CAM_INDEX, cv2.CAP_DSHOW)
# rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# if not rgb_cap.isOpened():
#     pipeline.stop()
#     raise RuntimeError("❌ External RGB camera NOT accessible")

# print("✅ External RGB camera opened")

# # =========================
# # MAIN LOOP (EXAMPLE)
# # =========================
# while True:

#     ret, rgb_frame = rgb_cap.read()
#     if not ret:
#         print("RGB frame grab failed")
#         break

#     cv2.imshow("External RGB Live", rgb_frame)

#     frames = pipeline.poll_for_frames()
#     if frames:
#         frames = align.process(frames)
#         depth = frames.get_depth_frame()
#         if depth:
#             pass  # depth available

#     key = cv2.waitKey(1) & 0xFF
#     if key == EXIT_KEY:
#         break

# # =========================
# # CLEANUP
# # =========================
# rgb_cap.release()
# pipeline.stop()
# cv2.destroyAllWindows()
