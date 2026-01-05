import cv2
import numpy as np
import pyrealsense2 as rs

# ==============================
# CONFIG
# ==============================
WIDTH, HEIGHT = 1280, 720
RGB_CAM_INDEX = 2
SAVE_KEY = ord('s')
EXIT_KEY = 27

# ==============================
# GLOBALS
# ==============================
H_global = None
rs_depth_warped_global = None
rs_intrinsics = None
depth_scale = None
common_bbox = None   # (x1, y1, x2, y2)

# ==============================
# MOUSE CALLBACK
# ==============================
def mouse_callback(event, x, y, flags, param):
    global H_global, rs_depth_warped_global, rs_intrinsics, depth_scale, common_bbox

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if H_global is None or rs_depth_warped_global is None:
        print("❌ Common FOV not computed yet")
        return

    x1, y1, x2, y2 = common_bbox

    if not (x1 <= x <= x2 and y1 <= y <= y2):
        print("❌ Click outside common FOV")
        return

    depth_raw = rs_depth_warped_global[y, x]
    if depth_raw == 0:
        print("❌ No depth at this pixel")
        return

    Z = depth_raw * depth_scale

    # Pixel → camera coordinate
    X = (x - rs_intrinsics.ppx) * Z / rs_intrinsics.fx
    Y = (y - rs_intrinsics.ppy) * Z / rs_intrinsics.fy

    print(f"📍 Pixel: ({x},{y}) → XYZ = ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")

# ==============================
# OPEN EXTERNAL RGB CAMERA
# ==============================
rgb_cap = cv2.VideoCapture(RGB_CAM_INDEX, cv2.CAP_DSHOW)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not rgb_cap.isOpened():
    raise RuntimeError("External RGB camera NOT accessible")

# ==============================
# INIT REALSENSE
# ==============================
ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise RuntimeError("No RealSense detected")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm-up
for _ in range(60):
    pipeline.wait_for_frames()

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

rs_intrinsics = profile.get_stream(rs.stream.color) \
    .as_video_stream_profile().get_intrinsics()

# ==============================
# FEATURE MATCHING
# ==============================
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cv2.namedWindow("External RGB Live Feed")
cv2.setMouseCallback("External RGB Live Feed", mouse_callback)

print("===================================")
print("LIVE RGB")
print("Press 'S' → Compute Common FOV")
print("Click → Get XYZ")
print("ESC → Exit")
print("===================================")

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, rgb_frame = rgb_cap.read()
    if not ret:
        break

    rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
    display = rgb_frame.copy()

    if common_bbox is not None:
        x1, y1, x2, y2 = common_bbox
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("External RGB Live Feed", display)
    key = cv2.waitKey(1) & 0xFF

    # ==========================
    # COMPUTE COMMON FOV
    # ==========================
    if key == SAVE_KEY:
        print("Computing common FOV...")

        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("RealSense frame error")
            continue

        rs_color = np.asanyarray(color_frame.get_data())
        rs_depth = np.asanyarray(depth_frame.get_data())

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

        rs_color_warped = cv2.warpPerspective(rs_color, H, (WIDTH, HEIGHT))
        rs_depth_warped = cv2.warpPerspective(rs_depth, H, (WIDTH, HEIGHT))

        mask = np.any(rs_color_warped > 0, axis=2)
        ys, xs = np.where(mask)

        if len(xs) < 5000:
            print("No common area")
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # SAVE GLOBALS
        H_global = H
        rs_depth_warped_global = rs_depth_warped
        common_bbox = (x1, y1, x2, y2)

        print("✅ Common FOV ready")

    elif key == EXIT_KEY:
        break

# ==============================
# CLEANUP
# ==============================
rgb_cap.release()
pipeline.stop()
cv2.destroyAllWindows()
