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

---

## ROS 2 TM Cobot Pipeline

The following scripts run as ROS 2 nodes for the **TM5-700** cobot using the TM driver, MoveIt, and the built-in TM camera.

### Prerequisites

1. Build the workspace:
   ```bash
   cd ~/cobot-pick-place
   colcon build --packages-select custom_package
   source install/setup.bash
   ```

2. Ensure the TM driver is running (or use the provided launch files which start it automatically).

---

### Step 1: Eye-in-Hand Calibration

Calibrates both camera intrinsics (using a 9×6 chessboard) and the camera-to-TCP extrinsic transform.

**Using the launch file (recommended):**
```bash
ros2 launch custom_package eye_in_hand_calibration.launch.py robot_ip:=192.168.1.2
```

**Or run the node directly** (if TM driver is already running):
```bash
ros2 run custom_package eye_in_hand_calibration.py
ros2 run custom_package eye_in_hand_calibration.py --ros-args -p square_size:=0.025 -p output:=eye_in_hand_calibration.npz
```

**Workflow:**

| Phase | What happens | Key |
|-------|-------------|-----|
| **Phase 1** — Intrinsic Calibration | Move the 9×6 chessboard in front of the TM camera at various angles. Frames auto-capture every 1s when corners are detected. | `c` = calibrate, `q` = quit |
| **Phase 2** — Eye-in-Hand Calibration (Automated) | The cobot automatically moves to 17 predefined poses via MoveIt. Keep the chessboard **fixed on the table**. At each pose, the system auto-detects the board and captures the pair. | `q` = abort |

**Configuration:** Edit `CALIB_CENTER_X/Y/Z` and `CALIB_POSE_OFFSETS` at the top of `eye_in_hand_calibration.py` to set the center position above your chessboard and the offset pattern.

**Output:** `eye_in_hand_calibration.npz` containing `mtx`, `dist`, and `T_tcp_to_camera`.

---

### Step 2: Pick-and-Place Demo

Runs the full autonomous continuous pick-and-place loop, executing a programmed multi-cube stacking sequence with visual servoing.

**Using the launch file (recommended):**
```bash
ros2 launch custom_package pick_place_demo.launch.py robot_ip:=192.168.1.2 calib_file:=eye_in_hand_calibration.npz
```

**Or run the node directly:**
```bash
ros2 run custom_package pick_place_demo.py
ros2 run custom_package pick_place_demo.py --ros-args -p calib_file:=eye_in_hand_calibration.npz
```

**Configuration:** Edit the constants at the top of `pick_place_demo.py` to match your setup:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKER_SIZE` | `0.02` | ArUco marker size (meters) |
| `CUBE_SIZE` | `0.03` | Cube dimensions (3cm) |
| `TCP_OFFSET_X/Y/Z` | `0, 0, 0.15` | Gripper offset from flange (meters) |
| `TOP_VIEW_X/Y/Z` | `0.35, 0, 0.45` | Top-view survey position (meters) |
| `TOP_VIEW_ROLL/PITCH/YAW` | `π, 0, 0` | Top-view orientation (radians) |
| `GRIPPER_PIN` | `0` | Digital output pin for gripper |
| `APPROACH_HEIGHT` | `0.08` | Height above target to approach from |
| `MOVE_VELOCITY` | `0.3` | MoveIt velocity scaling |

**Demo loop (Stacking Sequence):**
The script executes a looped sequence `[(1, 0), (2, 1), (2, 3), (1, 2)]` representing `(source, target)`. For each step:
1. Move to top-view → detect source and target cubes.
2. Approach source cube with visual servoing (fine-tuning target location) → descend → close gripper.
3. Retreat to safe travel height and translate to target cube.
4. Visual servoing (fine-tuning zero parallax location) over the target cube to assure precision.
5. Align gripper overhead → descend to stack offset → open gripper → vertical retreat.
6. Loop to the next stack pair in the sequence.

The camera window displays real-time ArUco detection with 3D axes, cube wireframe overlay, and XYZ/RPY text.

---

### Other ROS 2 Nodes

```bash
# Move the cobot to a specific XYZRPY coordinate
ros2 run custom_package move_xyzrpy.py

# View the TM camera stream
ros2 run custom_package sub_img

# Get real-time 6D pose of the TCP
ros2 run custom_package get_pose_tmros2

# Send a raw command to the TM robot
ros2 run custom_package tm_send_command
```

imaage_talker
move_group


