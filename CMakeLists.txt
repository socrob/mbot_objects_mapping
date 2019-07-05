cmake_minimum_required(VERSION 2.8.3)
project(mbot_objects_mapping)

find_package(catkin REQUIRED
COMPONENTS
    rospy
    mbot_perception_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
)

#catkin_install_python(
#  PROGRAMS ros/scripts/interactive_node.sh
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

