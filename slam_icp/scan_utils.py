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

"""Utilities to convert ``sensor_msgs/LaserScan`` data into 2D point clouds."""

import math
from typing import Union

import numpy as np

PoseLike = Union[tuple, list, np.ndarray, object]


def laserscan_to_points(scan, max_range: float = 10.0) -> np.ndarray:
    """
    Convert a ``LaserScan`` to ``(N, 2)`` points in the sensor frame.

    Beams that are non-finite or fall outside the usable range window are
    dropped, so ``N`` may be smaller than the number of beams.
    """
    ranges = np.asarray(scan.ranges, dtype=float)
    n = ranges.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=float)

    angle_min = float(scan.angle_min)
    angle_increment = float(scan.angle_increment)
    angles = angle_min + np.arange(n) * angle_increment

    range_min = float(getattr(scan, 'range_min', 0.0))
    scan_max = float(getattr(scan, 'range_max', max_range))
    upper = min(max_range, scan_max)

    valid = np.isfinite(ranges) & (ranges > range_min) & (ranges <= upper)
    r = ranges[valid]
    a = angles[valid]

    xs = r * np.cos(a)
    ys = r * np.sin(a)
    return np.column_stack((xs, ys))


def filter_valid_points(points: np.ndarray,
                        min_range: float = 0.1) -> np.ndarray:
    """Drop NaN/Inf points and any point closer than ``min_range`` to origin."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] == 0:
        return points.copy()
    finite = np.isfinite(points).all(axis=1)
    norms = np.linalg.norm(points, axis=1)
    keep = finite & (norms >= min_range)
    return points[keep]


def _pose_xytheta(pose: PoseLike):
    """Extract ``(x, y, yaw)`` from a tuple, Particle, or ``geometry_msgs/Pose``."""
    if isinstance(pose, (tuple, list, np.ndarray)):
        return float(pose[0]), float(pose[1]), float(pose[2])
    if hasattr(pose, 'yaw'):
        return float(pose.x), float(pose.y), float(pose.yaw)
    # geometry_msgs/Pose
    from slam_icp.util import yaw_from_quaternion
    yaw = yaw_from_quaternion(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return float(pose.position.x), float(pose.position.y), yaw


def transform_points(points: np.ndarray, pose: PoseLike) -> np.ndarray:
    """Transform points from the robot frame into the map frame given ``pose``."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] == 0:
        return points.copy()
    x, y, yaw = _pose_xytheta(pose)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T + np.array([x, y], dtype=float)
