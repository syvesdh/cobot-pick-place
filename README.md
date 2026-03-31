# Cobot Pick Place Vision System

This project provides a set of Python scripts for computer vision and robotics tasks, specifically generating ArUco markers, calibrating the camera, and performing 6D pose estimation.

## Installation

1.  Install the required dependencies:
    ```bash
    py -m pip install -r requirements.txt
    ```

## Usage

Run the script to generate markers. On Windows, it is recommended to use the `py` launcher:

```bash
py generate_aruco.py
```

### Options

You can customize the generation using command-line arguments:

-   `--num`: Number of markers to generate (default: 10).
-   `--size`: Size of each marker in pixels (default: 1000).
-   `--dict`: The ArUco dictionary to use (default: `DICT_4X4_50`).
-   `--dir`: Output directory for the images (default: `markers`).

Example:
```bash
py generate_aruco.py --num 10 --size 1000 --dict DICT_6X6_100 --dir my_markers
```

### Printing Tips

-   The generated markers include a **white border** to ensure they are easily detectable even when placed on dark surfaces.
-   When printing, ensure you do not use "Fit to Page" if you need specific physical dimensions. 
-   The default size (400px) is sufficient for most desktop printers and will result in a clear marker even at smaller physical sizes.
-   If using these for distance estimation, remember to measure the physical width of the marker (the black part) after printing.

## Camera Calibration

Calibrate your camera using an OpenCV 9x6 chessboard to ensure accurate pose estimation.

```bash
py calibrate_camera.py --square_size 0.0233
```

-   `--square_size`: Actual physical size of the chessboard squares in meters (e.g., `0.025` for 25mm).
-   `--cam`: Camera index to use (default: 0).
-   `--output`: File to save the parameters to (default: `camera_calibration.npz`).

Hold the chessboard in front of the camera at various angles and distances. The script automatically captures a frame every 1 second when the entire board is visible. Press `c` when you have accumulated enough frames (e.g. 20-30) to compute and save the calibration.

## ArUco 6D Pose Estimation

Track the 6D pose (Translation XYZ, Rotation RPY) of your specific ArUco markers.

```bash
py detect_aruco_pose.py --marker_size 0.02
```

-   `--marker_size`: Actual physical size of the ArUco marker in meters (e.g., `0.1` for 100mm).
-   `--cam`: Camera index to use (default: 0).
-   `--calib`: Path to the calibration parameters file (default: `camera_calibration.npz`).

The script looks for ArUco markers `ID 0` and `ID 1` from the `cv2.aruco.DICT_4X4_50` dictionary. It overlays a 3D coordinate frame on detected markers and calculates the true `XYZ` in meters and `Roll, Pitch, Yaw` in degrees. Press `q` to quit.
