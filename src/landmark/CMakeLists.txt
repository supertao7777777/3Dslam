cmake_minimum_required(VERSION 3.0.2)
project(landmark)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  # cartographer_ros_msgs
  visualization_msgs
  # param_space
)
 
add_message_files(
  FILES
  QRPose.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
 
catkin_package(

 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime

)
 
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
 


# # Define executable
# add_executable(tcp_server_node src/tcp_server_node.cpp)

# # Link libraries
# target_link_libraries(tcp_server_node
#   ${catkin_LIBRARIES}
# )

# # Install the executable
# install(TARGETS tcp_server_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )


# Define executable
add_executable(tcp_server_node src/tcp_server_node_msg.cpp)

# Link libraries
target_link_libraries(tcp_server_node
  ${catkin_LIBRARIES}
)

# Install the executable
install(TARGETS tcp_server_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)