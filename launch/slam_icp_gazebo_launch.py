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
Launch slam_icp on Gazebo Classic with a TurtleBot3 burger.

Boots Gazebo Classic with a configurable world, spawns the robot, starts
``robot_state_publisher`` and the ``slam_icp_node`` (Python SLAM), and an
optional RViz2 instance.

Topic contract:
    /scan, /odom (in)  ->  /map, /slam_pose, /particlecloud, /slam_path,
    /odom_path (out), TF map -> odom.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Build the launch description for the slam_icp Gazebo simulation."""
    pkg_share = get_package_share_directory('slam_icp')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz_slam.rviz')
    params_file = os.path.join(pkg_share, 'config', 'slam_icp.yaml')

    use_rviz = LaunchConfiguration('rviz', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    world = LaunchConfiguration(
        'world',
        default=PathJoinSubstitution([
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'turtlebot3_house.world',
        ]),
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world}.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch',
                'robot_state_publisher.launch.py',
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch',
                'spawn_turtlebot3.launch.py',
            )
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution([
                get_package_share_directory('turtlebot3_gazebo'),
                'worlds',
                'turtlebot3_house.world',
            ]),
        ),

        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        gazebo_launch,
        robot_state_publisher,
        spawn_turtlebot3,

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
