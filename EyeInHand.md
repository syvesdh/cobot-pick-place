# Hand-Eye Calibration for TM Cobot

To determine the camera frame relative to the Tool Center Point (TCP) in a TM cobot with an eye-in-hand setup, a process called Hand-Eye Calibration is required. 

Mathematically, this involves solving the spatial transformation equation:
AX = XB

Here, X is the transformation matrix from the TCP to the camera (T_tcp^cam) that needs to be found.

## Practical Steps

1. Prepare Calibration Target: Print a checkerboard or ChArUco board and fix it securely in the workspace. Ensure it does not move at all during the calibration process.
2. Collect Poses: Move the robot to about 10-15 different poses around the target. Make sure the orientation (rotation) varies significantly across the x, y, and z axes so the resulting matrix converges accurately.
3. Get Kinematic Data: At each pose, record the T_base^tcp value (the TCP pose relative to the robot base) directly from the TM controller.
4. Get Visual Data: Simultaneously at each pose, capture an image with the camera to get the T_cam^target value (the calibration target pose relative to the camera) using a pose estimation algorithm.
5. Calculate: Feed these collected pose pairs into a calibration algorithm, such as the Tsai-Lenz or Park-Martin methods.

## Implementation Notes

To save time and effort while migrating the robotics system to ROS Jazzy, the ROS ecosystem can be leveraged. Packages like `easy_handeye` wrap and automate this entire process. Alternatively, for a custom Python script, the built-in `cv2.calibrateHandEye` function from OpenCV is the standard approach to compute the matrix.