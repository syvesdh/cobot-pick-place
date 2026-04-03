cmake_minimum_required(VERSION 3.5)
project(custom_package)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(techman_robot_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tm_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(tm_send_command src/send_command.cpp)
  ament_target_dependencies(tm_send_command
  "techman_robot_msgs"
  "rclcpp"
  "tm_msgs"
)

add_executable(sub_img src/sub_img.cpp)
ament_target_dependencies(sub_img
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "ament_index_cpp"
)

add_executable(calibrate_camera_tmros2 src/calibrate_camera_tmros2.cpp)
ament_target_dependencies(calibrate_camera_tmros2
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "ament_index_cpp"
)

add_executable(detect_aruco_pose_tmros2 src/detect_aruco_pose_tmros2.cpp)
ament_target_dependencies(detect_aruco_pose_tmros2
  "rclcpp"
  "rclcpp_components"
  "sensor_msgs"
  "std_msgs"
  "OpenCV"
  "ament_index_cpp"
)

add_executable(get_pose_tmros2 src/get_pose_tmros2.cpp)
ament_target_dependencies(get_pose_tmros2
  "rclcpp"
  "tm_msgs"
)

install(TARGETS
  tm_send_command
  sub_img
  calibrate_camera_tmros2
  detect_aruco_pose_tmros2
  get_pose_tmros2
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY image
  DESTINATION share/${PROJECT_NAME}
)

ament_package()