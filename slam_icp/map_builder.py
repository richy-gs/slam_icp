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

"""Online 2D occupancy-grid builder using Bresenham raycasting + log-odds."""

from typing import List, Tuple

import numpy as np

from slam_icp.scan_utils import transform_points
from slam_icp.util import in_bounds, world_to_map


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Integer Bresenham line from ``(x0, y0)`` to ``(x1, y1)`` inclusive."""
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


class MapBuilder:
    """Maintain a log-odds occupancy grid fused from scans at known poses."""

    def __init__(self, resolution: float, width: int, height: int,
                 origin: Tuple[float, float],
                 l_occ: float = 0.85, l_free: float = -0.4,
                 l_clamp: float = 5.0):
        """Create an empty grid of ``width`` x ``height`` cells."""
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_clamp = float(l_clamp)

        self._log_odds = np.zeros((self.height, self.width), dtype=float)
        self._known = np.zeros((self.height, self.width), dtype=bool)

    def update(self, scan_points: np.ndarray, robot_pose) -> None:
        """Integrate a scan (points in ``base_frame``) taken at ``robot_pose``."""
        scan_points = np.asarray(scan_points, dtype=float).reshape(-1, 2)
        if hasattr(robot_pose, 'yaw'):
            rx, ry = robot_pose.x, robot_pose.y
            pose_xyt = (robot_pose.x, robot_pose.y, robot_pose.yaw)
        else:
            rx, ry = robot_pose[0], robot_pose[1]
            pose_xyt = (robot_pose[0], robot_pose[1], robot_pose[2])

        robot_i, robot_j = world_to_map(
            rx, ry, self.origin_x, self.origin_y, self.resolution)

        if scan_points.shape[0] == 0:
            return
        world_pts = transform_points(scan_points, pose_xyt)

        for px, py in world_pts:
            ei, ej = world_to_map(
                px, py, self.origin_x, self.origin_y, self.resolution)
            ray = bresenham(robot_i, robot_j, ei, ej)
            for ci, cj in ray[:-1]:
                if in_bounds(ci, cj, self.width, self.height):
                    self._apply(cj, ci, self.l_free)
            if in_bounds(ei, ej, self.width, self.height):
                self._apply(ej, ei, self.l_occ)

    def _apply(self, row: int, col: int, delta: float) -> None:
        value = self._log_odds[row, col] + delta
        self._log_odds[row, col] = float(
            np.clip(value, -self.l_clamp, self.l_clamp))
        self._known[row, col] = True

    def to_numpy(self) -> np.ndarray:
        """Return the grid as ``(H, W)`` int8 with values in ``{-1, 0..100}``."""
        prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
        grid = np.full((self.height, self.width), -1, dtype=np.int8)
        known = self._known
        grid[known] = np.round(prob[known] * 100.0).astype(np.int8)
        return grid

    def get_map(self):
        """Return a ``nav_msgs/OccupancyGrid`` of the current map (lazy import)."""
        from nav_msgs.msg import OccupancyGrid
        grid = self.to_numpy()
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.width)
        msg.info.height = int(self.height)
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().astype(np.int8).tolist()
        return msg
