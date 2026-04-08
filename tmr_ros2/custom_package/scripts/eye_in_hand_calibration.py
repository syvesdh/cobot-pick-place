#!/usr/bin/env python3
"""
Eye-in-Hand Calibration for TM5-700 Cobot
==========================================
Two-phase calibration:
  Phase 1: Camera intrinsic calibration (9x6 chessboard)
  Phase 2: Eye-in-hand extrinsic calibration (camera-to-TCP)
           The cobot is commanded via MoveIt to visit a set of
           calibration poses automatically.

Subscribes to:
  - techman_image   (sensor_msgs/Image)         — TM camera stream
  - feedback_states (tm_msgs/FeedbackState)      — real-time TCP pose

Calls:
  - move_action     (moveit_msgs/action/MoveGroup) — MoveIt PTP movement

Saves to a single .npz file:
  - mtx, dist                — camera intrinsics
  - T_tcp_to_camera (4x4)    — extrinsic transform

Usage:
  ros2 run custom_package eye_in_hand_calibration.py
  ros2 run custom_package eye_in_hand_calibration.py --ros-args -p square_size:=0.025 -p output:=eye_in_hand_calibration.npz
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions


import cv2
import numpy as np
import math
import time
import threading
import os


# ============================================================
# CONFIGURATION
# ============================================================
BOARD_SIZE = (9, 6)                 # Internal corners (width, height)
DEFAULT_SQUARE_SIZE = 0.006         # Chessboard square size in meters
DEFAULT_OUTPUT = "eye_in_hand_calibration.npz"
CAPTURE_INTERVAL = 1.0              # Seconds between auto-captures

# Eye-in-hand calibration poses (XYZ in meters, RPY in radians)
# The cobot will move to each of these poses automatically.
# The chessboard should be fixed on the table and visible from all poses.
# For good calibration: vary X,Y position and tilt angles.
# ==> EDIT THESE to match your workspace and chessboard location <==
CALIB_CENTER_X = 0.33               # X center above the chessboard
CALIB_CENTER_Y = 0.0                # Y center above the chessboard
CALIB_CENTER_Z = 0.50               # Z height above the chessboard
CALIB_LOOKING_DOWN_ROLL = 3.14159   # π — camera facing straight down
CALIB_LOOKING_DOWN_PITCH = 0.0
CALIB_LOOKING_DOWN_YAW = 1.5708

# Offsets from the center to create diverse viewpoints
# Format: (dx, dy, dz, d_roll, d_pitch, d_yaw) — added to center values
CALIB_POSE_OFFSETS = [
    # Straight down at different XY positions
    ( 0.00,  0.00,  0.00,   0.0,    0.0,    0.0  ),   # Center
    ( 0.05,  0.00,  0.00,   0.0,    0.0,    0.0  ),   # Right
    (-0.05,  0.00,  0.00,   0.0,    0.0,    0.0  ),   # Left
    ( 0.00,  0.05,  0.00,   0.0,    0.0,    0.0  ),   # Forward
    ( 0.00, -0.05,  0.00,   0.0,    0.0,    0.0  ),   # Backward
    # Tilted views
    ( 0.04,  0.00,  0.02,   0.0,    0.15,   0.0  ),   # Tilt pitch +
    (-0.04,  0.00,  0.02,   0.0,   -0.15,   0.0  ),   # Tilt pitch -
    ( 0.00,  0.04,  0.02,   0.15,   0.0,    0.0  ),   # Tilt roll +
    ( 0.00, -0.04,  0.02,  -0.15,   0.0,    0.0  ),   # Tilt roll -
    # Diagonal with tilt
    ( 0.04,  0.04,  0.02,   0.12,   0.12,   0.0  ),   # Diagonal NE
    (-0.04, -0.04,  0.02,  -0.12,  -0.12,   0.0  ),   # Diagonal SW
    ( 0.04, -0.04,  0.02,  -0.12,   0.12,   0.0  ),   # Diagonal SE
    (-0.04,  0.04,  0.02,   0.12,  -0.12,   0.0  ),   # Diagonal NW
    # Different heights
    ( 0.00,  0.00, -0.05,   0.0,    0.0,    0.0  ),   # Lower
    ( 0.00,  0.00,  0.05,   0.0,    0.0,    0.0  ),   # Higher
    # Yaw rotation
    ( 0.03,  0.03,  0.00,   0.0,    0.0,    0.15 ),   # Yaw +
    (-0.03, -0.03,  0.00,   0.0,    0.0,   -0.15 ),   # Yaw -
]

MOVE_VELOCITY = 0.3
MOVE_ACCELERATION = 0.3
SETTLE_TIME = 4.0   # Seconds to wait after moving before capturing


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def euler_to_quaternion(roll, pitch, yaw):
    """Convert RPY (radians) to quaternion [x, y, z, w]."""
    qx = (math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
          - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
    qy = (math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
          + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2))
    qz = (math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
          - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2))
    qw = (math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2)
          + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2))
    return [qx, qy, qz, qw]


def euler_xyz_deg_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """
    Convert TM robot Euler angles (Rx, Ry, Rz in degrees) to a 3x3 rotation matrix.
    TM uses XYZ extrinsic (= ZYX intrinsic) convention.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = np.array([
        [1,           0,            0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx),  math.cos(rx)]
    ])
    Ry = np.array([
        [ math.cos(ry), 0, math.sin(ry)],
        [            0, 1,            0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
def euler_xyz_to_rotation_matrix(rx, ry, rz):
    """TM Euler angles (Rx, Ry, Rz in radians) → 3x3 rotation matrix."""
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx),  math.cos(rx)]])
    Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz),  math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler_zyx(R):
    """Rotation matrix → (roll, pitch, yaw) in radians (ZYX convention)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
    return roll, pitch, yaw


def pose_to_homogeneous(x, y, z, rx, ry, rz):
    """TM tool_pose [meters, radians] → 4x4 homogeneous matrix [meters]."""
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_rotation_matrix(rx, ry, rz)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def build_calibration_poses():
    """Build the full list of calibration poses from center + offsets."""
    poses = []
    for dx, dy, dz, dr, dp, dyw in CALIB_POSE_OFFSETS:
        poses.append((
            CALIB_CENTER_X + dx,
            CALIB_CENTER_Y + dy,
            CALIB_CENTER_Z + dz,
            CALIB_LOOKING_DOWN_ROLL + dr,
            CALIB_LOOKING_DOWN_PITCH + dp,
            CALIB_LOOKING_DOWN_YAW + dyw,
        ))
    return poses


# ============================================================
# MAIN NODE
# ============================================================

class EyeInHandCalibration(Node):
    def __init__(self):
        super().__init__('eye_in_hand_calibration')

        # Parameters
        self.declare_parameter('square_size', DEFAULT_SQUARE_SIZE)
        self.declare_parameter('output', DEFAULT_OUTPUT)
        self.square_size = self.get_parameter('square_size').get_parameter_value().double_value
        self.output_file = self.get_parameter('output').get_parameter_value().string_value

        # Chessboard object points
        self.objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Storage for intrinsic calibration
        self.objpoints = []
        self.imgpoints = []
        self.frames_captured = 0
        self.last_capture_time = time.time()

        # Storage for eye-in-hand calibration
        self.R_gripper2base_list = []   # Rotation: base -> TCP
        self.t_gripper2base_list = []   # Translation: base -> TCP
        self.R_target2cam_list = []     # Rotation: camera -> board
        self.t_target2cam_list = []     # Translation: camera -> board
        self.hand_eye_captures = 0

        # Latest camera image and TCP pose
        self.latest_image = None
        self.display_image = None
        self.latest_tcp_pose = None
        self.last_key = -1
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        # Calibration results
        self.mtx = None
        self.dist = None
        self.img_size = None

        # Sub-pixel corner refinement criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # TM set_positions service client
        self.set_pos_client = self.create_client(SetPositions, 'set_positions')

        # ROS subscriptions
        self.image_sub = self.create_subscription(
            Image, 'techman_image', self.image_callback, 10)
        self.pose_sub = self.create_subscription(
            FeedbackState, 'feedback_states', self.pose_callback, 10)

        self.get_logger().info('Eye-in-Hand Calibration node started.')
        self.get_logger().info(f'  Square size: {self.square_size}m')
        self.get_logger().info(f'  Output file: {self.output_file}')
        self.get_logger().info(f'  Calibration poses: {len(CALIB_POSE_OFFSETS)}')
        
        self.running = True

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    def image_callback(self, msg):
        """Convert ROS Image to OpenCV BGR matrix."""
        try:
            if msg.encoding in ('rgb8',):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('bgr8', '8UC3'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'mono8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

            with self.image_lock:
                self.latest_image = frame.copy()
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def pose_callback(self, msg):
        """Store latest TCP pose from feedback_states."""
        if len(msg.tool_pose) == 6:
            with self.pose_lock:
                self.latest_tcp_pose = list(msg.tool_pose)

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    def move_to(self, x, y, z, roll, pitch, yaw):
        """Send a LINE_T goal via TM set_positions service (bypasses MoveIt IK flips). Returns True on success."""
        self.get_logger().info(
            f'Moving via LINE_T to XYZ=[{x:.4f}, {y:.4f}, {z:.4f}] RPY=[{roll:.3f}, {pitch:.3f}, {yaw:.3f}]')

        if not self.set_pos_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('set_positions service not available! Is TM driver running?')
            return False

        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.LINE_T
        req.positions = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        req.velocity = 0.2         # m/s
        req.acc_time = 0.2         # time to reach max speed (s)
        req.blend_percentage = 0   # 0%
        req.fine_goal = True       # precise position

        # Call service asynchronously, then wait
        future = self.set_pos_client.call_async(req)
        
        t_start = time.time()
        while not future.done() and (time.time() - t_start) < 10.0:
            time.sleep(0.05)

        if not future.done():
            self.get_logger().error('set_positions service call timed out!')
            return False

        result = future.result()
        if result and result.ok:
            self.get_logger().info('Move command queued successfully.')
            
            # Synchronously wait for the physical robot to reach the target block
            self.get_logger().info('Waiting for arm to physically reach target...')
            target_pos = np.array([float(x), float(y), float(z)])
            arm_arrived = False
            time.sleep(0.5)

            wait_start = time.time()
            while not arm_arrived and (time.time() - wait_start) < 20.0:
                with self.pose_lock:
                    curr_pose = self.latest_tcp_pose
                
                if curr_pose is not None:
                    curr_pos = np.array(curr_pose[:3])
                    # Euclidean distance in meters
                    dist = np.linalg.norm(curr_pos - target_pos) 
                    
                    if dist < 0.005:  # within 5mm
                        arm_arrived = True
                        break
                
                time.sleep(0.1)
                
            if not arm_arrived:
                self.get_logger().error('Movement timed out! Arm never arrived at the target coordinates.')
                return False
                
            return True
        else:
            self.get_logger().error('Move failed (service returned false).')
            return False

    # --------------------------------------------------------
    # Main Calibration Loop
    # --------------------------------------------------------

    def run_calibration(self):
        """Main calibration loop — runs in a background thread."""
        # Wait for first image
        self.get_logger().info('Waiting for camera image on /techman_image ...')
        while self.running and rclpy.ok():
            with self.image_lock:
                if self.latest_image is not None:
                    break
            time.sleep(0.1)

        if not self.running or not rclpy.ok():
            return

        self.get_logger().info('Camera image received! Starting Phase 1: Intrinsic Calibration')
        self.phase1_intrinsic()

        if not self.running or not rclpy.ok():
            return

        if self.mtx is not None:
            self.get_logger().info('Starting Phase 2: Eye-in-Hand Extrinsic Calibration')
            self.phase2_eye_in_hand()

        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

    # --------------------------------------------------------
    # Phase 1: Intrinsic Calibration
    # --------------------------------------------------------

    def phase1_intrinsic(self):
        """Phase 1: Collect chessboard images for intrinsic camera calibration."""
        print("=" * 60)
        print("PHASE 1: Camera Intrinsic Calibration")
        print("=" * 60)
        print("Move the 9x6 chessboard in front of the TM camera.")
        print("Frames are auto-captured every 1s when corners are detected.")
        print("  'c' — Calculate & save intrinsics (need 10+ frames)")
        print("  'q' — Quit without saving")
        print("=" * 60)

        while self.running and rclpy.ok():
            with self.image_lock:
                frame = self.latest_image.copy() if self.latest_image is not None else None

            if frame is None:
                time.sleep(0.03)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.img_size = gray.shape[::-1]

            ret_corners, corners = cv2.findChessboardCorners(
                gray, BOARD_SIZE, None,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

            display = frame.copy()

            if ret_corners:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, ret_corners)

                current_time = time.time()
                if current_time - self.last_capture_time >= CAPTURE_INTERVAL:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners2)
                    self.frames_captured += 1
                    self.last_capture_time = current_time
                    print(f"  [Phase 1] Captured frame {self.frames_captured}. Move the board!")
                    display = cv2.bitwise_not(display)

            cv2.putText(display, f"Phase 1: Intrinsic | Captured: {self.frames_captured}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to calibrate, 'q' to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            with self.image_lock:
                self.display_image = display

            key = self.last_key
            if key != -1:
                self.last_key = -1  # consume

            if key == ord('q'):
                print("Exiting Phase 1 without calibration.")
                self.running = False
                return
            elif key == ord('c'):
                if self.frames_captured < 5:
                    print(f"  Warning: Only {self.frames_captured} frames. Need at least 5.")
                else:
                    break
            
            time.sleep(0.05)

        if self.frames_captured > 0:
            print("\nCalculating camera intrinsics... Please wait.")
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.img_size, None, None)

            if ret > 0:
                print(f"Intrinsic calibration successful! RMS error: {ret:.4f}")
                print(f"\nCamera Matrix:\n{self.mtx}")
                print(f"\nDistortion Coefficients:\n{self.dist}")
            else:
                print("Intrinsic calibration failed.")
                self.mtx = None
                self.running = False
        else:
            print("No valid frames captured.")
            self.running = False

    # --------------------------------------------------------
    # Phase 2: Eye-in-Hand (Automated via MoveIt)
    # --------------------------------------------------------

    def phase2_eye_in_hand(self):
        """
        Phase 2: Automatically move the cobot to predefined poses via MoveIt,
        detect the chessboard at each pose, and collect pairs for eye-in-hand calibration.
        """
        calib_poses = build_calibration_poses()
        total_poses = len(calib_poses)

        print("\n" + "=" * 60)
        print("PHASE 2: Eye-in-Hand Extrinsic Calibration (Automated)")
        print("=" * 60)
        print(f"The cobot will automatically move to {total_poses} poses via MoveIt.")
        print("Keep the chessboard FIXED on the table — do not move it.")
        print("At each pose the system will detect the board and capture the pair.")
        print("")
        print("  'q' — Abort calibration")
        print("  's' — Skip current pose")
        print("=" * 60)

        # Wait for TM set_positions service
        self.get_logger().info('Waiting for TM set_positions service...')
        if not self.set_pos_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error('set_positions service not available! Cannot run Phase 2.')
            return

        for i, (x, y, z, r, p, yw) in enumerate(calib_poses):
            if not self.running or not rclpy.ok():
                break

            print(f"\n--- Pose {i + 1}/{total_poses} ---")

            # Move cobot to the calibration pose with retries
            attempts = 0
            success = False
            while attempts < 3 and not success:
                success = self.move_to(x, y, z, r, p, yw)
                if not success:
                    print(f"  [!] Move service failed. Retrying ({attempts + 1}/3)...")
                    time.sleep(1.0)
                attempts += 1

            if not success:
                print(f"  [!] Failed to reach pose {i + 1} permanently. Skipping this detection.")
                continue

            # Wait for the camera image to settle
            print(f"  Settling for {SETTLE_TIME}s...")
            time.sleep(SETTLE_TIME)

            # Try to detect chessboard and capture
            captured = self._try_capture_pair()

            if captured:
                print(f"  [Phase 2] Captured pair {self.hand_eye_captures} at pose {i + 1}.")
            else:
                print(f"  [!] No chessboard detected at pose {i + 1}. Skipping.")

            # Check for user abort/skip via OpenCV key
            key = self.last_key
            if key != -1:
                self.last_key = -1  # consume
                
            if key == ord('q'):
                print("User aborted Phase 2.")
                break

        print(f"\nTotal pairs captured: {self.hand_eye_captures}")

        # Compute eye-in-hand transform
        if self.hand_eye_captures >= 3:
            print("\nCalculating eye-in-hand transform... Please wait.")
            try:
                R_cam2tcp, t_cam2tcp = cv2.calibrateHandEye(
                    self.R_gripper2base_list,
                    self.t_gripper2base_list,
                    self.R_target2cam_list,
                    self.t_target2cam_list,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )

                T_tcp_to_camera = np.eye(4)
                T_tcp_to_camera[:3, :3] = R_cam2tcp
                T_tcp_to_camera[:3, 3] = t_cam2tcp.flatten()

                print(f"\nT_tcp_to_camera (4x4):\n{T_tcp_to_camera}")

                # Save everything
                np.savez(self.output_file,
                         mtx=self.mtx,
                         dist=self.dist,
                         T_tcp_to_camera=T_tcp_to_camera)
                print(f"\nSaved calibration to {self.output_file}")
                print("  Contains: mtx, dist, T_tcp_to_camera")

            except Exception as e:
                print(f"Eye-in-hand calibration failed: {e}")
        else:
            print(f"Not enough pairs ({self.hand_eye_captures}). Need at least 3 (10+ recommended).")
            print("Try adjusting CALIB_CENTER_X/Y/Z or CALIB_POSE_OFFSETS in the script.")

    def _try_capture_pair(self, max_attempts=10):
        """
        Attempt to detect the chessboard in the current camera frame
        and capture the board + TCP pose pair.
        Returns True if successful.
        """
        for attempt in range(max_attempts):
            with self.image_lock:
                frame = self.latest_image.copy() if self.latest_image is not None else None
            with self.pose_lock:
                tcp_pose = list(self.latest_tcp_pose) if self.latest_tcp_pose is not None else None

            if frame is None or tcp_pose is None:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.img_size is None:
                self.img_size = gray.shape[::-1]

            ret_corners, corners = cv2.findChessboardCorners(
                gray, BOARD_SIZE, None,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

            # Update display
            display = frame.copy()
            if ret_corners:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, ret_corners)

            if tcp_pose is not None:
                tcp_text = f"TCP: X={tcp_pose[0]:.1f} Y={tcp_pose[1]:.1f} Z={tcp_pose[2]:.1f} mm"
                cv2.putText(display, tcp_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            status = "BOARD DETECTED" if ret_corners else "Searching..."
            color = (0, 255, 0) if ret_corners else (0, 0, 255)
            cv2.putText(display,
                        f"Phase 2: Eye-in-Hand | Pairs: {self.hand_eye_captures} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "'q'=abort",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            with self.image_lock:
                self.display_image = display

            if not ret_corners:
                time.sleep(0.2)
                continue

            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # Solve camera -> board
            success, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.mtx, self.dist)
            if not success:
                time.sleep(0.2)
                continue

            # Build T_base_to_tcp from the current robot pose
            T_base_tcp = pose_to_homogeneous(*tcp_pose)
            R_g2b = T_base_tcp[:3, :3]
            t_g2b = T_base_tcp[:3, 3].reshape(3, 1)

            # Board -> Camera
            R_t2c, _ = cv2.Rodrigues(rvec)
            t_t2c = tvec.reshape(3, 1)

            self.R_gripper2base_list.append(R_g2b)
            self.t_gripper2base_list.append(t_g2b)
            self.R_target2cam_list.append(R_t2c)
            self.t_target2cam_list.append(t_t2c)
            self.hand_eye_captures += 1

            # Flash feedback
            display = cv2.bitwise_not(display)
            with self.image_lock:
                self.display_image = display
            time.sleep(0.3)
            return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = EyeInHandCalibration()
    
    # Spin ROS 2 in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Run the calibration math/sequence in its own background thread
    process_thread = threading.Thread(target=node.run_calibration, daemon=True)
    process_thread.start()
    
    try:
        # Run CV2 GUI safely strictly on the Main Thread (Required by Wayland/macOS)
        cv2.namedWindow('Eye-in-Hand Calibration', cv2.WINDOW_AUTOSIZE)
        while rclpy.ok() and node.running:
            with node.image_lock:
                frame = node.display_image.copy() if node.display_image is not None else None
            
            if frame is not None:
                cv2.imshow('Eye-in-Hand Calibration', frame)
                
            key = cv2.waitKey(30) & 0xFF
            if key != 255:  # OpenCV returns 255 if nothing is pressed
                node.last_key = key
                
            if key == ord('q') and node.last_key == -1: # if we want instant kill for saftey
                node.running = False
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
