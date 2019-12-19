cmake_minimum_required(VERSION 2.8.3)
project(stereo_loc)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE "Debug")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_filters
  cv_bridge
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)

catkin_package(
 INCLUDE_DIRS include/stereo_loc
#  LIBRARIES stereo_loc
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########


include_directories(
include/stereo_loc
${catkin_INCLUDE_DIRS}
"/usr/include/eigen3"
${Pangolin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(goal_viewer lib/goal_viewer.cpp)
target_link_libraries(goal_viewer
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

add_library(math_tools
  lib/math_tools.cpp
)

add_library(stereo_loc
  lib/stereo_loc.cpp
)
target_link_libraries(stereo_loc
  ${OpenCV_LIBS}
  goal_viewer
  math_tools
)

add_executable(location_node src/location_node.cpp)
target_link_libraries(location_node
  ${catkin_LIBRARIES}
  stereo_loc
)

add_executable(test_loc src/test_loc.cpp)
target_link_libraries(test_loc
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  stereo_loc
)

add_executable(fit_line src/fit_line.cpp)
target_link_libraries(fit_line
  ${OpenCV_LIBS}
)