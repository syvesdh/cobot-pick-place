# Running ROS 2 Camera Calibration via Docker

This guide walks you through compiling and running your newly-created `calibrate_camera_tmros2` node inside a Docker container.

> [!WARNING]
> You recently encountered the error `xhost: unable to open display ""`. This occurs because you are connected via an SSH session without an attached graphical display. 
> To enable OpenCV windows (`cv::imshow`) over SSH, you must disconnect and reconnect to `ninoxserver` with X11 forwarding enabled:
> ```bash
> ssh -X ubuntu@ninoxserver  # macOS / Linux clients
> ```
> *(Note: If you are connecting from a Windows machine using PowerShell/PuTTY, you will also need an X server like VcXsrv or Xming running locally in the background).*

## 1. Permit Local Docker Windows
Once you are logged in with a valid display variable, run this to permit the Docker container to draw windows to your forwarded display:
```bash
xhost +local:root
```

## 2. Launch the ROS 2 Container
Start your interactive ROS 2 container. The flags below ensure that camera streams and X11 configurations pass into the Docker instance.
```bash
docker run -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/ubuntu/cobot-pick-place:/home/ubuntu/cobot-pick-place \
    -w /home/ubuntu/cobot-pick-place/tmr_ros2 \
    osrf/ros:humble-desktop bash
```

## 3. Build the Package
Inside the Docker terminal context, compile the new CMake target utilizing standard ROS tooling.
```bash
sudo apt update && sudo apt install ros-humble-tf2-geometry-msgs -y
# Verify your core ROS 2 environment is active
source /opt/ros/humble/setup.bash

# Build the package
colcon build

# Source your freshly built local workspace
source install/setup.bash
```

## 4. Run the Camera Calibrator
Execute your new calibrator! It will automatically subscribe to your camera topics.
```bash
ros2 run custom_package calibrate_camera_tmros2
```

> [!TIP]
> Since this subscribes to the `techman_image` topic instead of opening `/dev/video0` directly, you MUST ensure that your core techman-robot ROS driver is running in another terminal so that it is actively publishing images across the shared `--net=host` network!
