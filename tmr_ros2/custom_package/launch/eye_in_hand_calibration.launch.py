"""
Launch file for Eye-in-Hand Calibration.

Launches:
  1. TM Driver (tm_driver) — connects to the TM5-700 and starts publishing
     techman_image + feedback_states
  2. eye_in_hand_calibration.py — the calibration node

Usage:
  ros2 launch custom_package eye_in_hand_calibration.launch.py
  ros2 launch custom_package eye_in_hand_calibration.launch.py robot_ip:=192.168.1.2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.2',
        description='IP address of the TM5-700 robot'
    )
    square_size_arg = DeclareLaunchArgument(
        'square_size',
        default_value='0.025',
        description='Chessboard square size in meters'
    )
    output_arg = DeclareLaunchArgument(
        'output',
        default_value='eye_in_hand_calibration.npz',
        description='Output calibration file path'
    )

    # TM Driver node
    tm_driver_node = Node(
        package='tm_driver',
        executable='tm_driver',
        name='tm_driver',
        output='screen',
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip'),
        }]
    )

    # Eye-in-Hand Calibration node
    calibration_node = Node(
        package='custom_package',
        executable='eye_in_hand_calibration.py',
        name='eye_in_hand_calibration',
        output='screen',
        parameters=[{
            'square_size': LaunchConfiguration('square_size'),
            'output': LaunchConfiguration('output'),
        }]
    )

    return LaunchDescription([
        robot_ip_arg,
        square_size_arg,
        output_arg,
        tm_driver_node,
        calibration_node,
    ])
