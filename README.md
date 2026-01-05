# RGB_Depth_Image_Calibration
RGB depth image calibration using two camera for better result.



**depth_measurement.py**
# RGB–Depth Camera Alignment and 3D Coordinate Extraction

This project implements a **human-vision–like fusion system** using two different cameras:

- **External RGB Camera** – for high-quality color imaging  
- **Intel RealSense D435i** – for accurate depth measurement  

The system aligns both camera views, detects their **common Field of View (FOV)**, and allows the user to **click on a pixel in the external RGB image** to obtain the corresponding **real-world 3D coordinates (X, Y, Z)**.

---

## 🔍 Key Features

- External RGB camera live streaming (high-quality color)
- Intel RealSense RGB + depth streaming
- Automatic **common FOV detection** using feature matching and homography
- Pixel mapping from **external RGB → RealSense RGB**
- Depth lookup from RealSense depth frame
- Real-world **3D coordinate (X, Y, Z in meters)** computation
- Interactive mouse click–based depth measurement
- Robust handling of camera conflicts on Windows

---

## 📐 Processing Pipeline

1. Capture live frames from:
   - External RGB camera (OpenCV)
   - RealSense RGB + Depth (pyrealsense2)
2. Compute **homography** between the two RGB views
3. Detect the **common FOV**
4. On mouse click:
   - Map RGB pixel → RealSense RGB pixel (via inverse homography)
   - Read depth value from RealSense depth frame
   - Convert pixel + depth → real-world XYZ coordinates

---

## 🧰 Requirements

### Hardware
- Intel RealSense D435 / D435i
- External USB RGB camera

### Software
- Python 3.8+
- Windows (recommended for DirectShow stability)

### Python Libraries
```bash
pip install opencv-python numpy pyrealsense2

⚠️ Ensure Intel RealSense SDK is installed correctly.

⚙️ Configuration
Edit these parameters in the script if needed:
WIDTH, HEIGHT = 1280, 720
RGB_CAM_INDEX = 2  # External RGB camera index
Use the correct camera index for your system.

▶️ How to Run
python depth_measurement.py

