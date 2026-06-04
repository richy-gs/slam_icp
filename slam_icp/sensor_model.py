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
Likelihood-field sensor model ported from ``mcl_cpp`` and adapted to SLAM.

In ``mcl_cpp`` the nearest-obstacle distance is computed once over a static
map. During SLAM the map changes, so the distance field is recomputed on
demand from the current occupancy grid via ``cv2.distanceTransform`` and
cached between updates (PRD section 13.6).
"""

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from slam_icp.scan_utils import laserscan_to_points, transform_points

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_TINY = -50.0


@dataclass
class SensorModelConfig:
    """Likelihood-field configuration (matches ``sensor_model`` YAML block)."""

    std_dev: float = 0.05
    max_beams: int = 30
    model: str = 'likelihood_field'


def gauss_pdf(x: float, mean: float, std_dev: float) -> float:
    """Gaussian PDF, equivalent to ``mcl_cpp``'s ``gauss_pdf``."""
    if std_dev <= 0.0:
        return 0.0
    z = (x - mean) / std_dev
    return math.exp(-0.5 * z * z) / (std_dev * _SQRT_2PI)


class LikelihoodFields:
    """Likelihood-field measurement model over a dynamic occupancy grid."""

    def __init__(self, std_dev: float = 0.05, max_beams: int = 30,
                 occupied_thresh: int = 50, max_range: float = 10.0):
        """Initialize the model with a Gaussian ``std_dev`` (meters)."""
        self.std_dev = float(std_dev)
        self.max_beams = int(max_beams)
        self.occupied_thresh = int(occupied_thresh)
        self.max_range = float(max_range)

        self._dist_field: Optional[np.ndarray] = None
        self._resolution = 0.05
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._width = 0
        self._height = 0
        self._has_field = False

    def compute_from_grid(self, grid: np.ndarray, resolution: float,
                          origin_x: float, origin_y: float) -> None:
        """
        Recompute the cached nearest-obstacle distance field (in meters).

        ``grid`` is an ``(H, W)`` occupancy array with values in ``{-1, 0..100}``
        (``row = j``, ``col = i``).
        """
        grid = np.asarray(grid)
        self._height, self._width = grid.shape
        self._resolution = float(resolution)
        self._origin_x = float(origin_x)
        self._origin_y = float(origin_y)

        obstacle = grid >= self.occupied_thresh
        if not obstacle.any():
            self._dist_field = None
            self._has_field = False
            return

        # Obstacles are zeros; distanceTransform yields the distance (in cells)
        # to the nearest zero pixel for every free cell.
        binary = np.where(obstacle, 0, 255).astype(np.uint8)
        dist_cells = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        self._dist_field = dist_cells.astype(float) * self._resolution
        self._has_field = True

    def _select_beams(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points
        if self.max_beams > 0 and points.shape[0] > self.max_beams:
            idx = np.linspace(0, points.shape[0] - 1, self.max_beams)
            return points[idx.astype(int)]
        return points

    def _beam_distances_from_points(self, points, pose) -> Optional[np.ndarray]:
        """Return per-beam nearest-obstacle distances for base-frame points."""
        if not self._has_field or self._dist_field is None:
            return None
        points = self._select_beams(np.asarray(points, dtype=float).reshape(-1, 2))
        if points.shape[0] == 0:
            return np.empty(0, dtype=float)

        world = transform_points(points, pose)
        cols = np.floor((world[:, 0] - self._origin_x) / self._resolution)
        rows = np.floor((world[:, 1] - self._origin_y) / self._resolution)
        cols = np.clip(cols, 0, self._width - 1).astype(int)
        rows = np.clip(rows, 0, self._height - 1).astype(int)
        return self._dist_field[rows, cols]

    def _beam_distances(self, scan, pose) -> Optional[np.ndarray]:
        """Return per-beam nearest-obstacle distances (meters) for ``pose``."""
        if not self._has_field or self._dist_field is None:
            return None
        points = laserscan_to_points(scan, self.max_range)
        return self._beam_distances_from_points(points, pose)

    def update(self, scan, pose, scan_points_base=None) -> float:
        """Return the linear particle weight (product of beam likelihoods)."""
        if scan is None and scan_points_base is None:
            return 1.0
        if scan_points_base is not None:
            dists = self._beam_distances_from_points(scan_points_base, pose)
        else:
            dists = self._beam_distances(scan, pose)
        if dists is None or dists.size == 0:
            return 1.0
        prob = 1.0
        for dist in dists:
            phit = gauss_pdf(dist, 0.0, self.std_dev)
            if phit >= 1.0:
                phit = 1.0
            prob *= phit
        return prob

    def log_update(self, scan, pose, scan_points_base=None) -> float:
        """Return the log particle weight, summed per beam (PRD section 13.9)."""
        if scan is None and scan_points_base is None:
            return 0.0
        if scan_points_base is not None:
            dists = self._beam_distances_from_points(scan_points_base, pose)
        else:
            dists = self._beam_distances(scan, pose)
        if dists is None or dists.size == 0:
            return 0.0
        log_w = 0.0
        for dist in dists:
            phit = gauss_pdf(dist, 0.0, self.std_dev)
            if phit >= 1.0:
                phit = 1.0
            log_w += math.log(phit) if phit > 0.0 else _LOG_TINY
        return log_w
