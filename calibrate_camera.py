import cv2
import numpy as np
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera using a 9x6 OpenCV chessboard.")
    parser.add_argument("--square_size", type=float, default=0.025, 
                        help="Actual size of the chessboard squares (default: 0.025 meters).")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--output", type=str, default="camera_calibration.npz", 
                        help="Output file for calibration parameters (default: camera_calibration.npz).")
    args = parser.parse_args()

    # Chessboard dimensions (internal corners - width, height)
    # The user asked for "9x6 opencv chessboard"
    board_size = (9, 6)
    square_size = args.square_size

    # Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    # Then scale by square_size to get real-world coordinates
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.cam}.")
        return

    print("=========================================================")
    print("Camera calibration started.")
    print("Moving the 9x6 chessboard in front of the camera.")
    print("The script will automatically capture a frame every 1 second")
    print("when it detects all the chessboard corners.")
    print("Press 'c' when you have captured enough frames (e.g., 20+).")
    print("Press 'q' at any time to quit without saving.")
    print("=========================================================")

    last_capture_time = time.time()
    frames_captured = 0
    capture_interval = 1.0 # 1 second

    # Optional sub-pixel corner refinement criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, board_size, None)

        display_frame = frame.copy()

        # If found, add object points, image points (after refining them)
        if ret_corners:
            # Draw and display the corners
            cv2.drawChessboardCorners(display_frame, board_size, corners, ret_corners)
            
            # Check interval
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                # Refine corners before saving
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                frames_captured += 1
                last_capture_time = current_time
                print(f"Captured frame {frames_captured}. Change the angle/position of the board.")
                
                # Visual flash effect to indicate capture
                display_frame = cv2.bitwise_not(display_frame)

        # Display info on screen
        cv2.putText(display_frame, f"Captured: {frames_captured}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'c' to calculate & save", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Camera Calibration', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting without calibration.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('c'):
            if frames_captured < 5:
                print(f"Warning: Only {frames_captured} frames captured. Calibration might be inaccurate. Need at least 5-10.")
                # We can still proceed if they really want to, but it's risky.
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    if frames_captured > 0:
        print("\nCalculating camera calibration... Please wait.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print(f"Calibration successful! RMS re-projection error: {ret:.4f}")
            print("\nCamera Matrix:\n", mtx)
            print("\nDistortion Coefficients:\n", dist)

            np.savez(args.output, mtx=mtx, dist=dist)
            print(f"\nSaved calibration parameters to {args.output}")
        else:
            print("Calibration failed.")
    else:
        print("No valid frames were captured.")

if __name__ == "__main__":
    main()
