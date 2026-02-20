cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  cv_bridge
  tf
  pcl_ros
  message_generation
)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(Eigen3 3.2 QUIET REQUIRED NO_MODULE)
set(GTSAM_DIR "/home/lsw/gtsam/build")
find_package(GTSAM REQUIRED QUIET)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)
# 添加CSF库的包含目录
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include(3rdparty/find_dependencies.cmake)

add_message_files(
  FILES
  cloud_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

set(teaser_src
  src/graph.cc
  src/teaser_utils/feature_matcher.cc
  src/teaser_utils/fpfh.cc
)

add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

find_package(Torch REQUIRED PATHS "/home/lsw/libtorch/share/cmake/Torch")

add_executable(alaserPGO 
  src/laserPosegraphOptimization.cpp
  src/terminator.cpp
  include/scancontext/Scancontext.cpp
  include/quatro/quatro.hpp
  ${teaser_src}
)
add_dependencies(alaserPGO ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_compile_options(alaserPGO 
  PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(alaserPGO 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
  pmc::pmc
)

add_executable(multirobotPGO 
  src/MultiRobotlaserPosegraphOptimization.cpp
  # src/MultiRobotlaserPosegraphOptimization_0902orin.cpp
  src/terminator.cpp
  src/quatroRegistration.cpp
  include/scancontext/Scancontext.cpp
  include/deepdesc/deepdesc.cpp
  src/CSF_filter.cpp
  ${teaser_src}
)
add_dependencies(multirobotPGO ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_compile_options(multirobotPGO 
  PRIVATE ${OpenMP_CXX_FLAGS}
)

target_include_directories(multirobotPGO PUBLIC ${TORCH_INCLUDE_DIRS}) 

target_link_libraries(multirobotPGO 
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  gtsam
  pmc::pmc
  CSF # 添加CSF库
)
