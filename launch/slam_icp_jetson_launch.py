# Copyright 2026 You
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Self-contained real-robot SLAM for Jetson (Puzzlebot + RPLidar).

Only ``slam_icp`` is required in the workspace. External drivers must provide:
    RPLidar  ->  /scan
    encoders ->  /VelocityEncL, /VelocityEncR

TF chain (single publisher per transform):
    map  ->  odom            slam_icp_node
    odom ->  base_footprint   wheel_odometry
    base_footprint -> laser    robot_state_publisher (bundled minimal URDF)

Usage (ROS 2 Humble):
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ros2 launch slam_icp slam_icp_jetson_launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Build the self-contained Jetson / real-robot launch description."""
    pkg_share = get_package_share_directory('slam_icp')
    params_file = os.path.join(pkg_share, 'config', 'slam_icp_jetson.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz_slam.rviz')
    urdf_path = os.path.join(pkg_share, 'urdf', 'puzzlebot_minimal.urdf')

    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    use_rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    wheel_radius = LaunchConfiguration('wheel_radius')
    wheel_base = LaunchConfiguration('wheel_base')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
        }],
    )

    wheel_odometry = Node(
        package='slam_icp',
        executable='wheel_odometry',
        name='wheel_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'wheel_radius': wheel_radius,
            'wheel_base': wheel_base,
        }],
    )

    scan_republisher = Node(
        package='slam_icp',
        executable='scan_republisher',
        name='scan_republisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_frame': 'laser',
        }],
    )

    slam_node = Node(
        package='slam_icp',
        executable='slam_icp_node',
        name='slam_icp_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/scan', '/scan_fixed')],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='log',
        arguments=['--display-config=' + rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'wheel_radius',
            default_value='0.05',
            description='Wheel radius (m)',
        ),
        DeclareLaunchArgument(
            'wheel_base',
            default_value='0.19',
            description='Track width between wheels (m)',
        ),
        robot_state_publisher,
        wheel_odometry,
        scan_republisher,
        slam_node,
        rviz_node,
    ])
