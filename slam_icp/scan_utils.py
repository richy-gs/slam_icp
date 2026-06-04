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
from typing import Optional, Tuple, Union

import numpy as np

from slam_icp.util import yaw_from_quaternion

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


def apply_planar_extrinsic(points: np.ndarray, dx: float, dy: float,
                           dyaw: float) -> np.ndarray:
    """Transform ``(N, 2)`` points from a sensor frame into a target frame."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] == 0:
        return points.copy()
    c = math.cos(dyaw)
    s = math.sin(dyaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T + np.array([dx, dy], dtype=float)


def planar_extrinsic_from_transform(transform) -> Tuple[float, float, float]:
    """Extract ``(dx, dy, dyaw)`` mapping source-frame points into target frame."""
    t = transform.transform
    yaw = yaw_from_quaternion(
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    return float(t.translation.x), float(t.translation.y), yaw


def lookup_scan_extrinsic(tf_buffer, target_frame: str, source_frame: str,
                          stamp, timeout_sec: float,
                          fallback: Optional[Tuple[float, float, float]] = None,
                          yaw_offset: float = 0.0):
    """
    Return ``(dx, dy, dyaw)`` from ``source_frame`` to ``target_frame``.

    Uses ``tf_buffer.lookup_transform(target, source, ...)``. On failure,
    returns ``fallback`` when provided.
    """
    if target_frame == source_frame:
        extrinsic = (0.0, 0.0, 0.0)
    else:
        extrinsic = None
        try:
            import rclpy
            from tf2_ros import TransformException

            if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
                when = rclpy.time.Time()
            else:
                when = stamp
            transform = tf_buffer.lookup_transform(
                target_frame, source_frame, when,
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
            extrinsic = planar_extrinsic_from_transform(transform)
        except Exception:  # noqa: BLE001 - TransformException or runtime import
            extrinsic = fallback

        if extrinsic is None:
            return None

    dx, dy, dyaw = extrinsic
    dyaw = math.atan2(math.sin(dyaw + yaw_offset),
                       math.cos(dyaw + yaw_offset))
    return dx, dy, dyaw


def laserscan_to_base_points(scan, max_range: float, tf_buffer,
                             base_frame: str, scan_frame: str,
                             timeout_sec: float,
                             min_range: float = 0.1,
                             use_extrinsic: bool = True,
                             fallback: Optional[Tuple[float, float, float]] = None,
                             yaw_offset: float = 0.0) -> np.ndarray:
    """Convert a scan to ``(N, 2)`` points expressed in ``base_frame``."""
    points = filter_valid_points(
        laserscan_to_points(scan, max_range), min_range)
    if not use_extrinsic or points.shape[0] == 0:
        return points

    source_frame = scan.header.frame_id or scan_frame
    extrinsic = lookup_scan_extrinsic(
        tf_buffer, base_frame, source_frame, scan.header.stamp,
        timeout_sec, fallback=fallback, yaw_offset=yaw_offset)
    if extrinsic is None:
        return points
    return apply_planar_extrinsic(points, *extrinsic)


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
