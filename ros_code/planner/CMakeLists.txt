cmake_minimum_required(VERSION 2.8.3)
project(planner)

# your C++ compilation flags
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# finding non-ROS libraries, e.g. Eigen
find_package (Eigen3 3 REQUIRED)

# finding ROS packages
find_package(catkin REQUIRED COMPONENTS
    # required for using ROS in C++
    roscpp 
    # other ros packages, e.g.
    std_msgs
    tf2
    tf2_ros
    # the local ROS package you want
    freicar_map
    raiscar_msgs
    freicar_common
    cv_bridge
)
# dependecies of the current package
catkin_package(CATKIN_DEPENDS 
    roscpp
    std_msgs
    tf2
    tf2_ros
    freicar_map
    raiscar_msgs
    freicar_common
    cv_bridge
)
# header files to consider
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# your executable ROS node
add_executable(${PROJECT_NAME}_node src/planner.cpp src/planner.h)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES}
                                           Eigen3::Eigen
)
