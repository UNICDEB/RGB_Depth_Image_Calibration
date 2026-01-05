import cv2
import numpy as np
import pyrealsense2 as rs

# ==============================
# CONFIG
# ==============================
WIDTH, HEIGHT = 1280, 720
SAVE_KEY = ord('s')
EXIT_KEY = 27

# ==============================
# INIT RGB CAMERA (External)
# ==============================
rgb_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not rgb_cap.isOpened():
    raise RuntimeError("External RGB camera not accessible")

# ==============================
# INIT REALSENSE (SAFE)
# ==============================
ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise RuntimeError("No RealSense device detected")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

# SAFE warm-up (NO wait_for_frames)
for _ in range(40):
    pipeline.poll_for_frames()

depth_scale = pipeline.get_active_profile() \
    .get_device().first_depth_sensor().get_depth_scale()

# ==============================
# FEATURE MATCHING
# ==============================
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

H_cached = None
mask_common = None
rs_color_warped = None
rs_depth_warped = None

print("====================================")
print("RGB + RealSense Alignment Running")
print("Press 'S' → save common FOV data")
print("Press ESC → exit")
print("====================================")

# ==============================
# MAIN LOOP
# ==============================
while True:
    # ---------- RGB CAMERA ----------
    ret, rgb_frame = rgb_cap.read()
    if not ret:
        print("RGB camera frame failed")
        break

    rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
    display_frame = rgb_frame.copy()

    # ---------- REALSENSE ----------
    frames = pipeline.poll_for_frames()
    if not frames:
        cv2.imshow("Aligned View", display_frame)
        if cv2.waitKey(1) == EXIT_KEY:
            break
        continue

    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        cv2.imshow("Aligned View", display_frame)
        if cv2.waitKey(1) == EXIT_KEY:
            break
        continue

    rs_color = np.asanyarray(color_frame.get_data())
    rs_depth = np.asanyarray(depth_frame.get_data())

    # ---------- FIND HOMOGRAPHY (ONCE) ----------
    if H_cached is None:
        gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        gray_rs = cv2.cvtColor(rs_color, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(gray_rgb, None)
        kp2, des2 = orb.detectAndCompute(gray_rs, None)

        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 25:
                src_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                dst_pts = np.float32(
                    [kp1[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

                if H is not None:
                    H_cached = H
                    print("Homography locked")

    # ---------- APPLY ALIGNMENT ----------
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

    # ---------- DISPLAY ----------
    cv2.imshow("Aligned View", display_frame)

    # ---------- SAVE ----------
    key = cv2.waitKey(1) & 0xFF

    if key == SAVE_KEY:
        if H_cached is None or mask_common is None:
            print("No common area found")
            continue

        ys, xs = np.where(mask_common)
        if len(xs) == 0:
            print("No common area found")
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        rgb_common = rgb_frame[y1:y2, x1:x2]
        rs_color_common = rs_color_warped[y1:y2, x1:x2]
        rs_depth_common = rs_depth_warped[y1:y2, x1:x2]

        cv2.imwrite("rgb_common.png", rgb_common)
        cv2.imwrite("rs_color_common.png", rs_color_common)

        with open("depth_common.txt", "w") as f:
            h, w = rs_depth_common.shape
            for y in range(h):
                for x in range(w):
                    d = rs_depth_common[y, x] * depth_scale
                    if d > 0:
                        f.write(f"{x},{y},{d:.4f}\n")

        print("Saved: RGB, RS color, Depth TXT (common FOV)")

    elif key == EXIT_KEY:
        break

# ==============================
# CLEANUP
# ==============================
rgb_cap.release()
pipeline.stop()
cv2.destroyAllWindows()
