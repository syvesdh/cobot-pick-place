"""
Launch file for Pick-and-Place Demo.

Launches:
  1. TM Driver (tm_driver) — robot connection + image/feedback publishers
  2. MoveIt (if needed) — for trajectory planning via move_action
  3. pick_place_demo.py — the full demo node

Usage:
  ros2 launch custom_package pick_place_demo.launch.py
  ros2 launch custom_package pick_place_demo.launch.py robot_ip:=192.168.1.2 calib_file:=eye_in_hand_calibration.npz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.2',
        description='IP address of the TM5-700 robot'
    )
    calib_file_arg = DeclareLaunchArgument(
        'calib_file',
        default_value='eye_in_hand_calibration.npz',
        description='Path to the eye-in-hand calibration file'
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

    # Try to include the TM5-700 MoveIt launch if available
    # This provides the /move_action action server
    try:
        tm5_moveit_dir = get_package_share_directory('tm5-700_moveit_config')
        moveit_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tm5_moveit_dir, 'launch', 'tm5-700_moveit_planning_execution.launch.py')
            )
        )
        use_moveit = True
    except Exception:
        moveit_launch = None
        use_moveit = False

    # Pick-Place Demo node
    demo_node = Node(
        package='custom_package',
        executable='pick_place_demo.py',
        name='pick_place_demo',
        output='screen',
        parameters=[{
            'calib_file': LaunchConfiguration('calib_file'),
        }]
    )

    actions = [
        robot_ip_arg,
        calib_file_arg,
        tm_driver_node,
    ]

    if use_moveit and moveit_launch:
        actions.append(moveit_launch)

    actions.append(demo_node)

    return LaunchDescription(actions)
