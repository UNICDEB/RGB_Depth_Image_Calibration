import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================
# CONFIG
# =====================================
WIDTH, HEIGHT = 1280, 720
SAVE_KEY = ord('s')
EXIT_KEY = 27

# =====================================
# INIT RGB CAMERA (External USB)
# =====================================
rgb_cap = cv2.VideoCapture(0)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not rgb_cap.isOpened():
    raise RuntimeError("RGB camera not opened")

# =====================================
# INIT REALSENSE
# =====================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

# Warm-up (VERY IMPORTANT)
for _ in range(30):
    pipeline.wait_for_frames()

# =====================================
# FEATURE MATCHING
# =====================================
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

H_cached = None  # store homography once found

print("===================================")
print("Live RGB + Depth Alignment Running")
print("Press 'S' to save common FOV images")
print("Press ESC to exit")
print("===================================")

# =====================================
# MAIN LOOP
# =====================================
while True:
    # -------- RGB CAMERA --------
    ret, rgb_frame = rgb_cap.read()
    if not ret:
        print("RGB camera read failed")
        break

    rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
    display_frame = rgb_frame.copy()

    # -------- REALSENSE --------
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        cv2.imshow("Aligned View", display_frame)
        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            break
        continue

    rs_color = np.asanyarray(color_frame.get_data())
    rs_depth = np.asanyarray(depth_frame.get_data())

    # -------- FIND HOMOGRAPHY (ONCE) --------
    if H_cached is None:
        gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        gray_rs = cv2.cvtColor(rs_color, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(gray_rgb, None)
        kp2, des2 = orb.detectAndCompute(gray_rs, None)

        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 20:
                src_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                dst_pts = np.float32(
                    [kp1[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

                if H is not None:
                    H_cached = H
                    print("Homography computed and locked")

    # -------- APPLY ALIGNMENT --------
    if H_cached is not None:
        rs_color_warped = cv2.warpPerspective(
            rs_color, H_cached, (WIDTH, HEIGHT)
        )

        rs_depth_warped = cv2.warpPerspective(
            rs_depth, H_cached, (WIDTH, HEIGHT)
        )

        mask_common = np.any(rs_color_warped > 0, axis=2)

        if np.count_nonzero(mask_common) > 5000:
            display_frame[mask_common] = cv2.addWeighted(
                rgb_frame[mask_common], 0.6,
                rs_color_warped[mask_common], 0.4, 0
            )

    # -------- DISPLAY --------
    cv2.imshow("Aligned View", display_frame)

    # -------- KEY HANDLING --------
    key = cv2.waitKey(1) & 0xFF

    if key == SAVE_KEY:
        if H_cached is not None:
            cv2.imwrite("rgb_common.png", rgb_frame)
            cv2.imwrite("depth_common.png", rs_depth_warped)
            print("Saved: rgb_common.png & depth_common.png")
        else:
            print("No common area found yet")

    elif key == EXIT_KEY:
        break

# =====================================
# CLEANUP
# =====================================
rgb_cap.release()
pipeline.stop()
cv2.destroyAllWindows()
