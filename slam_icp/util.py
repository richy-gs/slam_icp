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

"""Geometry and coordinate helpers ported from ``mcl_cpp/src/util.cpp``."""

import math
from typing import Tuple

import numpy as np


def angle_diff(angle1: float, angle2: float) -> float:
    """
    Return the signed shortest angular difference ``angle1 - angle2``.

    Ported verbatim from ``mcl_cpp::angle_diff``. The result lies in
    ``[-pi, pi]``.
    """
    d1 = angle1 - angle2
    d2 = 2.0 * math.pi - abs(d1)
    if d1 > 0.0:
        d2 *= -1.0
    if abs(d1) < abs(d2):
        return d1
    return d2


def normalize_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` using ``atan2`` for stability."""
    return math.atan2(math.sin(angle), math.cos(angle))


def euler_from_quaternion(x: float, y: float, z: float,
                          w: float) -> Tuple[float, float, float]:
    """Convert a quaternion into ``(roll, pitch, yaw)`` Euler angles."""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw component from a quaternion."""
    _, _, yaw = euler_from_quaternion(x, y, z, w)
    return yaw


def euler_to_quaternion(yaw: float, pitch: float = 0.0,
                        roll: float = 0.0) -> Tuple[float, float, float, float]:
    """Convert Euler angles to a quaternion ``(x, y, z, w)``."""
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Return a normalized planar quaternion ``(x, y, z, w)`` for ``yaw``."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def world_to_map(x: float, y: float, origin_x: float, origin_y: float,
                 resolution: float) -> Tuple[int, int]:
    """Convert world coordinates to integer map cell ``(col, row)``."""
    i = int(math.floor((x - origin_x) / resolution))
    j = int(math.floor((y - origin_y) / resolution))
    return i, j


def map_to_world(i: int, j: int, origin_x: float, origin_y: float,
                 resolution: float) -> Tuple[float, float]:
    """Convert a map cell ``(col, row)`` to the world coordinate of its center."""
    x = origin_x + (i + 0.5) * resolution
    y = origin_y + (j + 0.5) * resolution
    return x, y


def in_bounds(i: int, j: int, width: int, height: int) -> bool:
    """Return ``True`` when cell ``(i, j)`` is inside a ``width`` x ``height`` grid."""
    return 0 <= i < width and 0 <= j < height


def transform_matrix(dx: float, dy: float, dtheta: float) -> np.ndarray:
    """Build a 2D homogeneous transform from translation and rotation."""
    c = math.cos(dtheta)
    s = math.sin(dtheta)
    return np.array([
        [c, -s, dx],
        [s, c, dy],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def matrix_to_xytheta(transform: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 2D homogeneous transform into ``(dx, dy, dtheta)``."""
    dx = float(transform[0, 2])
    dy = float(transform[1, 2])
    dtheta = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    return dx, dy, dtheta
