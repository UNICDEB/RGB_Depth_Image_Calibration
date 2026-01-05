import cv2
import numpy as np

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize to same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Simple fusion (average)
    fused = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    cv2.imshow("Human-like Single View", fused)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
