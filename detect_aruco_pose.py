import cv2
import numpy as np
import argparse
import math
import os

def euler_angles_from_rotation_matrix(R):
    """
    Computes Euler angles (Roll, Pitch, Yaw) from a rotation matrix.
    Assumes ZYX ordering (Yaw, Pitch, Roll).
    Returns angles in degrees: (Roll, Pitch, Yaw).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # Convert to degrees
    return math.degrees(x), math.degrees(y), math.degrees(z)

def main():
    parser = argparse.ArgumentParser(description="Estimate 6D pose of ArUco markers 0 and 1.")
    parser.add_argument("--marker_size", type=float, default=0.1, 
                        help="Actual size of the ArUco marker (default: 0.1 meters).")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--calib", type=str, default="camera_calibration.npz", 
                        help="Path to the camera calibration file (default: camera_calibration.npz).")
    args = parser.parse_args()

    if not os.path.exists(args.calib):
        print(f"Error: Calibration file '{args.calib}' not found.")
        print("Please run calibrate_camera.py first to generate this file.")
        return

    # Load calibration data
    with np.load(args.calib) as data:
        mtx = data['mtx']
        dist = data['dist']

    print(f"Loaded camera matrix and distortion coefficients from {args.calib}")

    # Set up ArUco Dictionary and Parameters
    # As requested, checking for dictionary cv2.aruco.DICT_4X4_50
    dict_type = cv2.aruco.DICT_4X4_50
    
    # Handle OpenCV 4.7+ vs older versions
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_old_api = False
    except AttributeError:
        # Fallback for older OpenCV
        aruco_dict = cv2.aruco.Dictionary_get(dict_type)
        parameters = cv2.aruco.DetectorParameters_create()
        use_old_api = True

    # 3D points of the marker corners (top-left, top-right, bottom-right, bottom-left) in marker frame
    # Z axis is 0, standing on the plane, looking forward = +Z, Right = +X, Down = +Y
    m_half = args.marker_size / 2.0
    obj_points = np.array([
        [-m_half, -m_half, 0],
        [ m_half, -m_half, 0],
        [ m_half,  m_half, 0],
        [-m_half,  m_half, 0]
    ], dtype=np.float32)

    # Open camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.cam}.")
        return

    print("=========================================================")
    print(f"ArUco Pose Estimation started (Marker size: {args.marker_size}m).")
    print("Looking for markers with ID 0 and 1...")
    print("Press 'q' to quit.")
    print("=========================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if not use_old_api:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        display_frame = frame.copy()

        if ids is not None and len(ids) > 0:
            # Draw borders of the markers
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                
                # Check for ID 0 and 1
                if marker_id in [0, 1]:
                    marker_corners = corners[i][0]

                    # Solve PnP to get rotation and translation vectors
                    success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, mtx, dist)

                    if success:
                        # Draw 3D axis on the marker
                        cv2.drawFrameAxes(display_frame, mtx, dist, rvec, tvec, args.marker_size * 0.5)

                        # Extract translation
                        tx, ty, tz = tvec.flatten()

                        # Extract rotation (RPY)
                        rmat, _ = cv2.Rodrigues(rvec)
                        roll, pitch, yaw = euler_angles_from_rotation_matrix(rmat)

                        # Overlay text for each marker
                        info_text_pos = f"ID {marker_id} XYZ: {tx:.2f}, {ty:.2f}, {tz:.2f}"
                        info_text_rot = f"RPY: {roll:.1f}, {pitch:.1f}, {yaw:.1f}"

                        # Determine position to draw text based on the corner of the marker
                        text_x = int(marker_corners[0][0])
                        text_y = int(marker_corners[0][1]) - 20

                        # Make sure text doesn't go off-screen at the top
                        text_y = max(text_y, 40)

                        cv2.putText(display_frame, info_text_pos, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_frame, info_text_rot, (text_x, text_y + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Print to console as well
                        print(f"Marker {marker_id} -> XYZ: [{tx:.3f}, {ty:.3f}, {tz:.3f}], RPY: [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]")

        cv2.imshow("ArUco 6D Pose Estimation", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
