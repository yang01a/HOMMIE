cmake_minimum_required(VERSION 3.0.2)
project(random_patrol_package)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  move_base_msgs
  actionlib
)

## Declare ROS messages, services and actions if needed
# add_message_files(
#   FILES
# )
# add_service_files(
#   FILES
# )
# add_action_files(
#   FILES
# )

## Generate added messages and services
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

## Catkin specific configuration
catkin_package()

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Python executables
catkin_install_python(PROGRAMS
  src/random_patrol.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

