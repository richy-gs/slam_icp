#!/usr/bin/env bash
# Jetson + slam_icp on-robot stack — ROS 2 Humble dependencies.
#
# Native Humble apt packages target Ubuntu 22.04 (JetPack 6). On Ubuntu 20.04
# (JetPack 5) install Humble via Docker (ros:humble-ros-base) or upgrade the
# board image before running this script.
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
UBUNTU_VERSION="$(lsb_release -rs 2>/dev/null || echo unknown)"

if [[ "${ROS_DISTRO}" != "humble" ]]; then
  echo "Warning: slam_icp is developed for ROS 2 Humble; ROS_DISTRO=${ROS_DISTRO}" >&2
fi

if [[ "${UBUNTU_VERSION}" == "20.04" ]]; then
  echo "Note: Ubuntu 20.04 (JetPack 5) has no official ros-humble debs." >&2
  echo "      Use JetPack 6 (22.04), or run Humble in Docker — see docs/jetson_jetpack5_ubuntu20.md" >&2
fi

sudo apt update
sudo apt install -y \
  "ros-${ROS_DISTRO}-rclpy" \
  "ros-${ROS_DISTRO}-sensor-msgs" \
  "ros-${ROS_DISTRO}-nav-msgs" \
  "ros-${ROS_DISTRO}-geometry-msgs" \
  "ros-${ROS_DISTRO}-std-msgs" \
  "ros-${ROS_DISTRO}-tf2-ros" \
  "ros-${ROS_DISTRO}-tf2-geometry-msgs" \
  "ros-${ROS_DISTRO}-robot-state-publisher" \
  "ros-${ROS_DISTRO}-rviz2" \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-numpy \
  python3-scipy \
  python3-opencv \
  python3-yaml

echo ""
echo "Done (ROS 2 ${ROS_DISTRO}). Next steps on the Jetson:"
echo "  mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src"
echo "  # copy or clone slam_icp here (only this package is required)"
echo "  cd ~/ros2_ws"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  rosdep install -i --from-path src --rosdistro ${ROS_DISTRO} -y"
echo "  colcon build --packages-select slam_icp"
echo "  source install/setup.bash"
echo "  ros2 launch slam_icp slam_icp_jetson_launch.py"
