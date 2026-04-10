#!/usr/bin/env python3
"""
Pick-and-Place Demo for TM5-700 Cobot
======================================
Full autonomous pipeline:
  1. Load eye-in-hand calibration (intrinsics + T_tcp_to_camera)
  2. Move to top view, detect ArUco cubes (ID 0 and ID 1)
  3. Pick cube 1, place it on top of cube 0
  4. Loop

Subscribes to:
  - techman_image     (sensor_msgs/Image)         — TM camera stream
  - feedback_states   (tm_msgs/FeedbackState)      — real-time TCP pose

Calls:
  - move_action       (moveit_msgs/action/MoveGroup) — MoveIt PTP movement
  - set_io            (tm_msgs/srv/SetIO)            — Gripper digital I/O

Usage:
  ros2 run custom_package pick_place_demo.py
  ros2 run custom_package pick_place_demo.py --ros-args -p calib_file:=eye_in_hand_calibration.npz
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetIO
from tm_msgs.srv import SetPositions

import cv2
import numpy as np
import math
import time
import threading
import os


# ============================================================
# CONFIGURATION — Edit these values for your setup
# ============================================================

# Calibration file path (from eye_in_hand_calibration.py)
DEFAULT_CALIB_FILE = "eye_in_hand_calibration.npz"

# ArUco settings
MARKER_SIZE = 0.02            # ArUco marker size in meters (2cm)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Cube dimensions (meters)
CUBE_SIZE = 0.03              # 3cm cube
GRIP_DEPTH = -0.010           # 1cm depth offset into the cube

# Tool Center Point offset from flange to gripper tip (meters)
# Adjust these values to perfectly align your physical gripper tip
TCP_OFFSET_X = 0.10985
TCP_OFFSET_Y = -0.0763
TCP_OFFSET_Z = 0.10977

# Top-view survey position (XYZ in meters, RPY in radians)
# The cobot moves here to look down and detect both cubes
TOP_VIEW_X = 0.350
TOP_VIEW_Y = 0.15
TOP_VIEW_Z = 0.500
TOP_VIEW_ROLL = 3.14159       # π  — looking straight down
TOP_VIEW_PITCH = 0.0
TOP_VIEW_YAW = 1.5708

# Gripper I/O settings (Digital Output)
GRIPPER_MODULE = 0             # MODULE_CONTROLBOX = 0
GRIPPER_TYPE = 1               # TYPE_DIGITAL_OUT = 1
GRIPPER_PIN = 2                # Digital output pin number
GRIPPER_CLOSE_STATE = 1.0      # STATE_ON  = close gripper
GRIPPER_OPEN_STATE = 0.0       # STATE_OFF = open gripper

# Motion parameters
APPROACH_HEIGHT = 0.08         # Height above target to approach from (meters)
RETREAT_HEIGHT = 0.06          # Height to lift after grab (meters)
PLACE_STACK_OFFSET = 0.03     # Extra height for stacking (= cube height)
MOVE_VELOCITY = 0.3            # MoveIt velocity scaling (0.0–1.0)
MOVE_ACCELERATION = 0.3        # MoveIt acceleration scaling (0.0–1.0)

# Detection settling
DETECTION_FRAMES = 5           # Number of consistent detections before accepting


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


def draw_cube_on_marker(image, rvec, tvec, mtx, dist, marker_size):
    """
    Draw a 3D cube wireframe on an ArUco marker.
    The cube sits on top of the marker with height = marker_size.
    """
    half = marker_size / 2.0
    h = marker_size  # cube height

    # 8 corners of the cube (marker frame: Z points out of marker)
    cube_points = np.float32([
        [-half, -half, 0], [ half, -half, 0], [ half,  half, 0], [-half,  half, 0],  # bottom
        [-half, -half, -h], [ half, -half, -h], [ half,  half, -h], [-half,  half, -h]  # top
    ])

    img_pts, _ = cv2.projectPoints(cube_points, rvec, tvec, mtx, dist)
    img_pts = np.int32(img_pts).reshape(-1, 2)

    # Bottom face
    cv2.drawContours(image, [img_pts[:4]], -1, (0, 255, 0), 2)
    # Top face
    cv2.drawContours(image, [img_pts[4:]], -1, (0, 128, 255), 2)
    # Pillars
    for i in range(4):
        cv2.line(image, tuple(img_pts[i]), tuple(img_pts[i + 4]), (255, 0, 0), 2)


# ============================================================
# MAIN NODE
# ============================================================

class PickPlaceDemo(Node):
    def __init__(self):
        super().__init__('pick_place_demo')

        # Parameters
        self.declare_parameter('calib_file', DEFAULT_CALIB_FILE)
        self.calib_file = self.get_parameter('calib_file').get_parameter_value().string_value

        # Load calibration
        if not os.path.exists(self.calib_file):
            self.get_logger().error(f"Calibration file '{self.calib_file}' not found!")
            self.get_logger().error("Run eye_in_hand_calibration.py first.")
            raise FileNotFoundError(self.calib_file)

        calib = np.load(self.calib_file)
        self.mtx = calib['mtx']
        self.dist = calib['dist']
        self.T_tcp_to_camera = calib['T_tcp_to_camera']
        self.get_logger().info(f"Loaded calibration from {self.calib_file}")
        self.get_logger().info(f"Camera Matrix:\n{self.mtx}")
        self.get_logger().info(f"T_tcp_to_camera:\n{self.T_tcp_to_camera}")

        # ArUco setup
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
            params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        # ArUco marker 3D object points
        m_half = MARKER_SIZE / 2.0
        self.marker_obj_points = np.array([
            [-m_half, -m_half, 0],
            [ m_half, -m_half, 0],
            [ m_half,  m_half, 0],
            [-m_half,  m_half, 0]
        ], dtype=np.float32)

        # State
        self.latest_image = None
        self.display_image = None
        self.latest_tcp_pose = None
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.marker_lock = threading.Lock()

        # Flag for main thread shutdown
        self.running = True

        # Detected marker poses (in camera frame) — updated every frame
        self.detected_markers = {}  # {id: (rvec, tvec)} in camera frame

        # TM set_positions service client
        self.set_pos_client = self.create_client(SetPositions, 'set_positions')

        # SetIO service client (gripper)
        self.io_client = self.create_client(SetIO, 'set_io')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, 'techman_image', self.image_callback, 10)
        self.pose_sub = self.create_subscription(
            FeedbackState, 'feedback_states', self.pose_callback, 10)

        self.get_logger().info('Pick-and-Place Demo node started.')
        self.get_logger().info(f'  Marker size: {MARKER_SIZE}m')
        self.get_logger().info(f'  Cube size:   {CUBE_SIZE}m')
        self.get_logger().info(f'  TCP offset:  [{TCP_OFFSET_X}, {TCP_OFFSET_Y}, {TCP_OFFSET_Z}]')

        # Run main demo loop in background
        self.running = True
        self.demo_thread = threading.Thread(target=self.run_demo, daemon=True)
        self.demo_thread.start()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    def image_callback(self, msg):
        """Receive camera image, detect ArUco markers, display overlay."""
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

            # Detect and draw ArUco markers on display
            self._detect_and_display(frame)
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def _detect_and_display(self, frame):
        """Detect ArUco markers, draw overlays, update shared state."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        display = frame.copy()
        detected = {}

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id in (0, 1):
                    marker_corners = corners[i][0]
                    success, rvec, tvec = cv2.solvePnP(
                        self.marker_obj_points, marker_corners, self.mtx, self.dist)

                    if success:
                        # Draw 3D axis
                        cv2.drawFrameAxes(display, self.mtx, self.dist, rvec, tvec,
                                          MARKER_SIZE * 0.5)
                        # Draw 3D cube edges
                        draw_cube_on_marker(display, rvec, tvec, self.mtx, self.dist,
                                            MARKER_SIZE)

                        # Pose text overlay
                        tx, ty, tz = tvec.flatten()
                        rmat, _ = cv2.Rodrigues(rvec)
                        roll, pitch, yaw = rotation_matrix_to_euler_zyx(rmat)

                        text_x = int(marker_corners[0][0])
                        text_y = max(int(marker_corners[0][1]) - 30, 50)

                        cv2.putText(display,
                                    f"ID {marker_id} XYZ: {tx:.3f}, {ty:.3f}, {tz:.3f}",
                                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 255, 0), 2)
                        cv2.putText(display,
                                    f"RPY: {roll:.3f}, {pitch:.3f}, {yaw:.3f}",
                                    (text_x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (0, 255, 255), 2)

                        detected[marker_id] = (rvec, tvec)

        with self.marker_lock:
            self.detected_markers = detected

        # Safely hand off the rendered frame for the Main Thread to display
        with self.image_lock:
            self.display_image = display

    def pose_callback(self, msg):
        """Store latest TCP pose from robot feedback."""
        if len(msg.tool_pose) == 6:
            with self.pose_lock:
                self.latest_tcp_pose = list(msg.tool_pose)

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    def move_to(self, x, y, z, roll, pitch, yaw, wait=True):
        """Send a LINE_T goal via TM set_positions service (bypasses MoveIt IK flips)."""
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
            if wait:
                self.get_logger().info('Waiting for arm to physically reach target...')
                target_pos = np.array([float(x), float(y), float(z)])
                arm_arrived = False
                time.sleep(0.5)

                wait_start = time.time()
                last_curr_pos = None
                stationary_count = 0

                while not arm_arrived and (time.time() - wait_start) < 15.0:
                    with self.pose_lock:
                        curr_pose = self.latest_tcp_pose

                    if curr_pose is not None:
                        curr_pos = np.array(curr_pose[:3])
                        dist = np.linalg.norm(curr_pos - target_pos)
                        
                        # Check if arm has physically stopped moving
                        if last_curr_pos is not None:
                            movement = np.linalg.norm(curr_pos - last_curr_pos)
                            if movement < 0.0005:  # Moved less than 0.5mm
                                stationary_count += 1
                            else:
                                stationary_count = 0
                        last_curr_pos = curr_pos

                        # Arrived if it reached perfectly, or if it has completely stopped moving 
                        # for 0.5 seconds while reasonably close to the target coordinate
                        if dist < 0.005 or (stationary_count >= 5 and dist < 0.25):
                            self.get_logger().info(f'Arm reached target (Dist: {dist:.4f}m, Stationary checks: {stationary_count}).')
                            arm_arrived = True
                            break
                    time.sleep(0.1)

                if arm_arrived:
                    # Brief settle to ensure vibration stops
                    time.sleep(0.5)
                else:
                    self.get_logger().warn('Arm moved but destination tolerance check timed out.')
                    return False
            return True
        else:
            self.get_logger().error('Move failed (service returned false).')
            return False

    # --------------------------------------------------------
    # Gripper
    # --------------------------------------------------------

    def set_gripper(self, close=True):
        """Open or close the gripper via SetIO service. Retries up to 3 times."""
        state = GRIPPER_CLOSE_STATE if close else GRIPPER_OPEN_STATE
        action = "CLOSE" if close else "OPEN"

        for attempt in range(1, 4):
            self.get_logger().info(f'Gripper {action} (Attempt {attempt}/3)...')

            if not self.io_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error('SetIO service not available!')
            else:
                request = SetIO.Request()
                request.module = GRIPPER_MODULE
                request.type = GRIPPER_TYPE
                request.pin = GRIPPER_PIN
                request.state = state

                future = self.io_client.call_async(request)
                
                # Asynchronous wait to prevent nested node spinning
                t_start = time.time()
                while not future.done() and (time.time() - t_start) < 5.0 and self.running and rclpy.ok():
                    time.sleep(0.05)

                if not future.done():
                    self.get_logger().error(f'Gripper {action} service call timed out!')
                else:
                    result = future.result()
                    if result and result.ok:
                        self.get_logger().info(f'Gripper {action} OK.')
                        return True
                    else:
                        self.get_logger().error(f'Gripper {action} failed (service returned false)!')

            if attempt < 3:
                self.get_logger().warn(f'Retrying gripper {action} in 500ms...')
                time.sleep(0.5)

        self.get_logger().error(f'Gripper {action} permanently failed after 3 attempts!')
        return False

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    def get_marker_pose_in_base(self, marker_id, attempts=30, settle_frames=DETECTION_FRAMES):
        """
        Wait until marker is detected consistently, then return its pose in base frame.
        Returns (x, y, z) in meters in the robot base frame, or None.
        """
        self.get_logger().info(f'Detecting marker {marker_id}...')
        consistent_count = 0
        last_pos = None

        for _ in range(attempts):
            time.sleep(0.2)

            with self.marker_lock:
                markers = dict(self.detected_markers)
            with self.pose_lock:
                tcp_pose = list(self.latest_tcp_pose) if self.latest_tcp_pose else None

            if marker_id not in markers or tcp_pose is None:
                consistent_count = 0
                continue

            rvec, tvec = markers[marker_id]

            # marker pose in camera frame
            T_cam_marker = np.eye(4)
            R_cm, _ = cv2.Rodrigues(rvec)
            T_cam_marker[:3, :3] = R_cm
            T_cam_marker[:3, 3] = tvec.flatten()

            # current TCP → base
            T_base_tcp = pose_to_homogeneous(*tcp_pose)

            # camera → base = base_tcp * tcp_camera
            T_base_camera = T_base_tcp @ self.T_tcp_to_camera

            # marker → base
            T_base_marker = T_base_camera @ T_cam_marker

            pos = T_base_marker[:3, 3]

            if last_pos is not None and np.linalg.norm(pos - last_pos) < 0.005:
                consistent_count += 1
            else:
                consistent_count = 1

            last_pos = pos.copy()

            if consistent_count >= settle_frames:
                R_base_marker = T_base_marker[:3, :3]
                _, _, marker_yaw = rotation_matrix_to_euler_zyx(R_base_marker)
                
                self.get_logger().info(
                    f'Marker {marker_id} in base frame: '
                    f'X={pos[0]*1000:.1f}mm Y={pos[1]*1000:.1f}mm Z={pos[2]*1000:.1f}mm Yaw={math.degrees(marker_yaw):.1f}°')
                return pos, marker_yaw

        self.get_logger().warn(f'Could not get stable detection for marker {marker_id}')
        return None, None

    def get_flange_target(self, pos_base, z_padding, target_yaw=TOP_VIEW_YAW):
        """
        Given the target cube position in Base frame, calculate where the FLANGE 
        needs to travel so the Gripper Tip perfectly reaches the target.
        """
        # Flange Rotation Matrix at target orientation
        R_f = euler_xyz_to_rotation_matrix(TOP_VIEW_ROLL, TOP_VIEW_PITCH, target_yaw)
        
        # Gripper Tip offset in Flange's local coordinate system
        tip_offset_local = np.array([TCP_OFFSET_X, TCP_OFFSET_Y, TCP_OFFSET_Z])
        
        # Rotate that local offset into the global Base frame
        tip_offset_base = R_f @ tip_offset_local
        
        # Flange absolute target = Target Pos - Offset
        flange_target = pos_base - tip_offset_base
        
        # Add the Z approach padding
        flange_target[2] += z_padding
        
        # Calculate visual TCP goal for logging
        tcp_goal = pos_base.copy()
        tcp_goal[2] += z_padding
        
        self.get_logger().info(f'TCP Goal:      X={tcp_goal[0]:.4f}, Y={tcp_goal[1]:.4f}, Z={tcp_goal[2]:.4f}')
        self.get_logger().info(f'Flange Target: X={flange_target[0]:.4f}, Y={flange_target[1]:.4f}, Z={flange_target[2]:.4f}')
        
        return flange_target

    def execute_move(self, x, y, z, roll, pitch, yaw, step_name="Move"):
        """Wrapper for move_to with automatic retry logic."""
        attempts = 0
        success = False
        while attempts < 15 and not success and self.running and rclpy.ok():
            success = self.move_to(x, y, z, roll, pitch, yaw)
            if not success:
                self.get_logger().warn(f"[{step_name}] Move failed. Retrying (Attempt {attempts + 1}/15)...")
                time.sleep(1.0)
            attempts += 1
        return success

    # --------------------------------------------------------
    # Demo Pipeline
    # --------------------------------------------------------

    def run_demo(self):
        """Main pick-and-place loop."""
        # Wait for subscriptions
        self.get_logger().info('Waiting for camera image and TCP feedback...')
        while self.running and rclpy.ok():
            with self.image_lock:
                has_img = self.latest_image is not None
            with self.pose_lock:
                has_pose = self.latest_tcp_pose is not None
            if has_img and has_pose:
                break
            time.sleep(0.2)

        if not self.running:
            return

        self.get_logger().info('=' * 60)
        self.get_logger().info('PICK-AND-PLACE DEMO STARTING')
        self.get_logger().info('=' * 60)

        # Open gripper initially
        self.set_gripper(close=False)
        time.sleep(0.5)

        cycle = 0
        while self.running and rclpy.ok():
            cycle += 1
            self.get_logger().info(f'\n--- Cycle {cycle} ---')

            # Step 1: Go to top view
            self.get_logger().info('Step 1: Moving to top view...')
            if not self.execute_move(TOP_VIEW_X, TOP_VIEW_Y, TOP_VIEW_Z,
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, TOP_VIEW_YAW, "Step 1"):
                self.get_logger().error('Failed to move to top view permanently. Retrying cycle...')
                time.sleep(2)
                continue

            time.sleep(1.0)  # Let camera settle

            # Step 2: Detect cube 1 and cube 0
            self.get_logger().info('Step 2: Detecting cubes...')
            pos_cube1, yaw_cube1 = self.get_marker_pose_in_base(1)
            pos_cube0, yaw_cube0 = self.get_marker_pose_in_base(0)

            if pos_cube1 is None:
                self.get_logger().warn('Cube 1 not detected! Retrying cycle...')
                time.sleep(2)
                continue
            if pos_cube0 is None:
                self.get_logger().warn('Cube 0 not detected! Retrying cycle...')
                time.sleep(2)
                continue
                
            # Normalize yaw to the nearest 90-degree face to minimize TM wrist joint rotation
            # A square cube is symmetrical every 90 degrees (pi/2).
            # This logic strictly bounds the correction between -45° to +45° (-pi/4 to +pi/4).
            yaw_correction_1 = (yaw_cube1 + math.pi/4) % (math.pi/2) - math.pi/4
            yaw_correction_0 = (yaw_cube0 + math.pi/4) % (math.pi/2) - math.pi/4
                
            yaw_1 = TOP_VIEW_YAW + yaw_correction_1
            yaw_0 = TOP_VIEW_YAW + yaw_correction_0

            # Step 3: Move above cube 1 (approach)
            self.get_logger().info('Step 3: Approaching cube 1...')
            approach_pos = self.get_flange_target(pos_cube1, APPROACH_HEIGHT, yaw_1)
            if not self.execute_move(approach_pos[0], approach_pos[1], approach_pos[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_1, "Step 3"):
                self.get_logger().error('Failed to approach cube 1 permanently.')
                continue

            # Step 3.5: Visual Servoing (Fine-tune with Zero Parallax)
            time.sleep(1.0)  # Let camera settle mechanically
            self.get_logger().info('Step 3.5: Fine-tuning cube 1 position...')
            fine_pos_cube1, fine_yaw_cube1 = self.get_marker_pose_in_base(1, settle_frames=3)
            if fine_pos_cube1 is not None:
                yaw_correction_1 = (fine_yaw_cube1 + math.pi/4) % (math.pi/2) - math.pi/4
                yaw_1 = TOP_VIEW_YAW + yaw_correction_1
                pos_cube1 = fine_pos_cube1
                self.get_logger().info('Successfully fine-tuned cube 1 position.')
            else:
                self.get_logger().warn('Could not fine-tune cube 1. Trusting initial coordinates.')

            # Step 4: Move down to grab position
            self.get_logger().info('Step 4: Descending to grab cube 1...')
            grab_pos = self.get_flange_target(pos_cube1, GRIP_DEPTH, yaw_1)
            if not self.execute_move(grab_pos[0], grab_pos[1], grab_pos[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_1, "Step 4"):
                self.get_logger().error('Failed to descend to cube 1 permanently.')
                continue

            # Step 5: Close gripper
            self.get_logger().info('Step 5: Grabbing cube 1...')
            self.set_gripper(close=True)
            time.sleep(0.8)

            # Step 6: Retreat up directly
            self.get_logger().info('Step 6: Retreating...')
            retreat_pos = self.get_flange_target(pos_cube1, RETREAT_HEIGHT, yaw_1)
            if not self.execute_move(retreat_pos[0], retreat_pos[1], retreat_pos[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_1, "Step 6"):
                self.get_logger().error('Failed to retreat after grab permanently.')
                continue

            # Step 6.5: Ascend to safe travel height (clears tallest stack)
            self.get_logger().info('Step 6.5: Ascending to travel height...')
            place_approach_pos = self.get_flange_target(pos_cube0, PLACE_STACK_OFFSET + APPROACH_HEIGHT, yaw_0)
            travel_z = max(retreat_pos[2], place_approach_pos[2])
            
            if not self.execute_move(retreat_pos[0], retreat_pos[1], travel_z,
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_1, "Step 6.5"):
                self.get_logger().error('Failed to ascend to safe travel height.')
                continue

            # Step 7: Move laterally above cube 0
            self.get_logger().info('Step 7: Moving laterally above cube 0...')
            if not self.execute_move(place_approach_pos[0], place_approach_pos[1], travel_z,
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_0, "Step 7"):
                self.get_logger().error('Failed to translate to cube 0 permanently.')
                continue

            # Step 7.5: Descend to place approach height
            if not self.execute_move(place_approach_pos[0], place_approach_pos[1], place_approach_pos[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_0, "Step 7.5"):
                self.get_logger().error('Failed to descend to approach height.')
                continue

            # Step 7.8: Visual Servoing (Fine-tune stack location with Zero Parallax)
            time.sleep(1.0)  # Settle camera
            self.get_logger().info('Step 7.8: Fine-tuning cube 0 stack location...')
            fine_pos_cube0, fine_yaw_cube0 = self.get_marker_pose_in_base(0, settle_frames=3)
            if fine_pos_cube0 is not None:
                yaw_correction_0 = (fine_yaw_cube0 + math.pi/4) % (math.pi/2) - math.pi/4
                yaw_0 = TOP_VIEW_YAW + yaw_correction_0
                pos_cube0 = fine_pos_cube0
                self.get_logger().info('Successfully fine-tuned cube 0 position.')
            else:
                self.get_logger().warn('Could not fine-tune cube 0. Trusting initial coordinates.')

            # Step 8: Move down to place position (on top of cube 0)
            self.get_logger().info('Step 8: Placing cube 1 on cube 0...')
            place_pos = self.get_flange_target(pos_cube0, PLACE_STACK_OFFSET, yaw_0)
            if not self.execute_move(place_pos[0], place_pos[1], place_pos[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_0, "Step 8"):
                self.get_logger().error('Failed to descend to place position permanently.')
                continue

            # Step 9: Open gripper
            self.get_logger().info('Step 9: Releasing cube 1...')
            self.set_gripper(close=False)
            time.sleep(0.8)

            # Step 10: Final retreat away from stack
            self.get_logger().info('Step 10: Final retreat...')
            # Recalculate retreat strictly relative to fine-tuned orientation
            place_approach_pos_fine = self.get_flange_target(pos_cube0, PLACE_STACK_OFFSET + APPROACH_HEIGHT, yaw_0)
            if not self.execute_move(place_approach_pos_fine[0], place_approach_pos_fine[1], place_approach_pos_fine[2],
                                     TOP_VIEW_ROLL, TOP_VIEW_PITCH, yaw_0, "Step 10"):
                self.get_logger().error('Failed to perform final retreat permanently.')
                continue

            self.get_logger().info(f'--- Cycle {cycle} complete! ---')
            time.sleep(2)

        self.get_logger().info('Demo finished.')
        self.running = False
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PickPlaceDemo()
        
        # Spin ROS 2 in a background daemon thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        # Start the pick-and-place logic in its own background daemon thread
        pipeline_thread = threading.Thread(target=node.run_demo, daemon=True)
        pipeline_thread.start()

        # Safely run CV2 GUI rendering strictly on the Main Python Thread
        window_name = 'Pick-Place Demo — ArUco Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Draw a placeholder quickly so Wayland doesn't mistakenly flag the window as unresponsive
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Waiting for Camera Stream...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, dummy_frame)
        cv2.waitKey(1)

        while rclpy.ok() and node.running:
            with node.image_lock:
                frame = node.display_image.copy() if node.display_image is not None else None
            
            if frame is not None:
                cv2.imshow(window_name, frame)
            
            # Using 1ms instead of 30ms prevents Wayland message queues from overflowing
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                node.running = False
                break

    except FileNotFoundError:
        print("Calibration file not found. Run eye_in_hand_calibration.py first.")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
