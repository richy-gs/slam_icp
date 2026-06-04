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

"""Pose graph with loop-closure detection and Levenberg-Marquardt optimization."""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from slam_icp.icp import icp
from slam_icp.particle import Particle
from slam_icp.scan_utils import transform_points
from slam_icp.util import matrix_to_xytheta, transform_matrix


@dataclass
class Edge:
    """A relative-pose constraint between two graph nodes."""

    i: int
    j: int
    transform: np.ndarray
    information: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float))


class PoseGraph:
    """Graph of robot poses connected by relative-motion constraints."""

    def __init__(self):
        """Create an empty pose graph."""
        self.nodes: List[Particle] = []
        self.edges: List[Edge] = []

    def add_node(self, pose: Particle) -> int:
        """Append a node for ``pose`` and return its integer id."""
        self.nodes.append(
            Particle(pose.x, pose.y, pose.yaw, getattr(pose, 'weight', 1.0)))
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, transform: np.ndarray,
                 information: Optional[np.ndarray] = None) -> None:
        """Add a relative constraint: ``pose_i^{-1} . pose_j == transform``."""
        if information is None:
            information = np.eye(3, dtype=float)
        self.edges.append(
            Edge(i=i, j=j,
                 transform=np.asarray(transform, dtype=float),
                 information=np.asarray(information, dtype=float)))

    def detect_loop_closure(
        self,
        current_scan: np.ndarray,
        scan_history: List[np.ndarray],
        min_distance: float = 0.5,
        icp_threshold: float = 0.1,
        min_node_skip: int = 20,
        max_correspondence_dist: float = 0.5,
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        Search older nodes for a geometric match to the current scan.

        Returns ``(candidate_id, T)`` where ``T`` maps the current scan into the
        candidate node's frame, or ``None`` when no loop closure is found.
        """
        if not self.nodes:
            return None
        current = self.nodes[-1]
        current_id = len(self.nodes) - 1
        limit = current_id - min_node_skip

        for j in range(0, max(0, limit)):
            cand = self.nodes[j]
            dist = math.hypot(current.x - cand.x, current.y - cand.y)
            if dist >= min_distance:
                continue
            if j >= len(scan_history):
                continue
            transform, rmse, success = icp(
                current_scan, scan_history[j],
                max_correspondence_dist=max_correspondence_dist)
            if success and rmse < icp_threshold:
                return j, transform
        return None

    def optimize(self, method: str = 'lm') -> List[Particle]:
        """
        Optimize all node poses to satisfy the edge constraints.

        Node 0 is held fixed as the anchor. Returns the optimized node list.
        """
        n = len(self.nodes)
        if n < 2 or not self.edges:
            return list(self.nodes)

        anchor = self.nodes[0]
        x0 = np.array(
            [v for p in self.nodes[1:] for v in (p.x, p.y, p.yaw)],
            dtype=float)

        edges = self.edges

        def unpack(params: np.ndarray) -> List[np.ndarray]:
            mats = [transform_matrix(anchor.x, anchor.y, anchor.yaw)]
            for k in range(n - 1):
                px, py, pth = params[3 * k:3 * k + 3]
                mats.append(transform_matrix(px, py, pth))
            return mats

        def residuals(params: np.ndarray) -> np.ndarray:
            mats = unpack(params)
            res = []
            for edge in edges:
                m_i = mats[edge.i]
                m_j = mats[edge.j]
                meas_inv = np.linalg.inv(edge.transform)
                err = meas_inv @ np.linalg.inv(m_i) @ m_j
                dx, dy, dtheta = matrix_to_xytheta(err)
                weights = np.sqrt(np.clip(np.diag(edge.information), 0.0, None))
                res.extend([weights[0] * dx, weights[1] * dy,
                            weights[2] * dtheta])
            return np.asarray(res, dtype=float)

        num_residuals = 3 * len(edges)
        solver = method if num_residuals >= x0.size else 'trf'
        result = least_squares(residuals, x0, method=solver)

        optimized = [Particle(anchor.x, anchor.y, anchor.yaw, anchor.weight)]
        for k in range(n - 1):
            px, py, pth = result.x[3 * k:3 * k + 3]
            optimized.append(Particle(px, py, pth, self.nodes[k + 1].weight))
        self.nodes = optimized
        return list(self.nodes)

    def reprojected_points(
            self, scan_history: List[np.ndarray]) -> np.ndarray:
        """Re-project all stored scans with the current node poses (world frame)."""
        chunks = []
        for k, pose in enumerate(self.nodes):
            if k >= len(scan_history):
                break
            chunks.append(transform_points(
                scan_history[k], (pose.x, pose.y, pose.yaw)))
        if not chunks:
            return np.empty((0, 2), dtype=float)
        return np.vstack(chunks)
