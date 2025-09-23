cmake_minimum_required(VERSION 3.11)
project(grand_tour_ros1)

#==
# Dependencies
#==

find_package(catkin REQUIRED)

#==
# Package Configuration
#==

catkin_package()

#==
# Package Configuration
#==

catkin_package()

catkin_install_python(
  PROGRAMS
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#==
# Package installation
#==


install(
  PROGRAMS  scripts/rosbag_play.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)