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

"""Particle dataclass ported from ``mcl_cpp/include/mcl_cpp/particle.hpp``."""

from dataclasses import dataclass

from slam_icp.util import quaternion_from_yaw, yaw_from_quaternion


@dataclass
class Particle:
    """A weighted 2D pose hypothesis ``(x, y, yaw)`` for the particle filter."""

    x: float
    y: float
    yaw: float
    weight: float = 1.0

    def to_pose_msg(self):
        """Return a ``geometry_msgs/Pose`` for this particle (lazy ROS import)."""
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.x = float(self.x)
        pose.position.y = float(self.y)
        pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(self.yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    @classmethod
    def from_pose_msg(cls, pose, weight: float = 1.0) -> 'Particle':
        """Build a particle from a ``geometry_msgs/Pose``."""
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return cls(x=pose.position.x, y=pose.position.y, yaw=yaw, weight=weight)
