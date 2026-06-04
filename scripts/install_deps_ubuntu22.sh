#!/usr/bin/env bash
# Install ROS 2 Humble dependencies for slam_icp on Ubuntu 22.04 (desktop sim).
set -euo pipefail

if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
  echo "Source ROS 2 Humble first: source /opt/ros/humble/setup.bash" >&2
  exit 1
fi

sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-numpy \
  python3-scipy \
  python3-opencv \
  python3-yaml \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-turtlebot3-gazebo \
  ros-humble-turtlebot3-description \
  ros-humble-rviz2

echo "Done. Clone slam_icp into your workspace src/, then:"
echo "  rosdep install -i --from-path src --rosdistro humble -y"
echo "  colcon build --packages-select slam_icp"
