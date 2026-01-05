import cv2

def find_external_camera(max_index=5):
    working = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                working.append(i)
            cap.release()

    return working
print("Searching for external cameras...")
cameras = find_external_camera()
print(f"Available camera indices: {cameras}")   