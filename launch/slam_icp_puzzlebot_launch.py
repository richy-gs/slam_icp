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
Launch slam_icp on Gazebo (GZ/Ignition) with a Puzzlebot.

Boots GZ Sim with a configurable Puzzlebot world, spawns a Puzzlebot equipped
with a LiDAR (``puzzlebot_jetson_lidar_ed``), brings up its
``robot_state_publisher`` and ros_gz bridges, starts the ``slam_icp_node``
(Python SLAM) and an optional RViz2 instance.

Unlike ``slam_icp_gazebo_launch.py`` (TurtleBot3 on Gazebo Classic), this uses
the ``puzzlebot_gazebo`` package, which runs on GZ Sim via ``ros_gz_sim``.

Requires the companion workspace to be built and sourced:
    src/TE3003B_Eq2/ros2_ws  (provides puzzlebot_gazebo + puzzlebot_description)

Topic contract:
    /scan, /odom (in)  ->  /map, /slam_pose, /particlecloud, /slam_path,
    /odom_path (out), TF map -> odom.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Build the launch description for the slam_icp Puzzlebot simulation."""
    pkg_share = get_package_share_directory('slam_icp')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz_slam.rviz')
    params_file = os.path.join(pkg_share, 'config', 'slam_icp.yaml')

    puzzlebot_share = get_package_share_directory('puzzlebot_gazebo')

    use_rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw = LaunchConfiguration('yaw')
    world = LaunchConfiguration('world')
    robot = LaunchConfiguration('robot')
    pause = LaunchConfiguration('pause')
    gui = LaunchConfiguration('gui')
    spawn_delay = LaunchConfiguration('robot_spawn_delay')

    # Bring up GZ Sim + world + /clock and /tf bridges.
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_share, 'launch', 'gazebo_world_launch.py')
        ),
        launch_arguments={
            'world': world,
            'pause': pause,
            'gui': gui,
            'verbosity': '4',
        }.items(),
    )

    # Spawn the Puzzlebot (robot_state_publisher + create + ros_gz bridges).
    # Delayed so Gazebo's /clock is available before spawning.
    spawn_puzzlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_share, 'launch',
                         'gazebo_puzzlebot_launch.py')
        ),
        launch_arguments={
            'robot': robot,
            'robot_name': '',
            'x': x_pose,
            'y': y_pose,
            'yaw': yaw,
            'prefix': '',
            'lidar_frame': 'laser_frame',
            'camera_frame': 'camera_link_optical',
            'tof_frame': 'tof_link',
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'world',
            default_value='obstacle_avoidance_4.world',
            description='Gazebo world file (from puzzlebot_gazebo/worlds)',
        ),
        DeclareLaunchArgument(
            'robot',
            default_value='puzzlebot_jetson_lidar_ed',
            description='Puzzlebot variant; must have a LiDAR for /scan',
        ),
        DeclareLaunchArgument(
            'pause',
            default_value='false',
            description='Start Gazebo paused',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Launch the Gazebo GUI',
        ),
        DeclareLaunchArgument(
            'robot_spawn_delay',
            default_value='2.0',
            description='Seconds to wait for Gazebo /clock before spawning',
        ),

        gazebo_launch,
        TimerAction(period=spawn_delay, actions=[spawn_puzzlebot]),

        Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            arguments=['--display-config=' + rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz),
        ),
        Node(
            package='slam_icp',
            executable='slam_icp_node',
            name='slam_icp_node',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
    ])
