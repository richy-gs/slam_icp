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

"""Tests for the pose graph and loop-closure optimization."""

import math

import numpy as np

from slam_icp.particle import Particle
from slam_icp.pose_graph import PoseGraph
from slam_icp.util import matrix_to_xytheta, transform_matrix


def _rel(a, b):
    """Relative transform ``T`` such that ``pose_a . T == pose_b``."""
    return np.linalg.inv(transform_matrix(*a)) @ transform_matrix(*b)


def _cloud(seed=0, n=80):
    """Return a reproducible 2D point cloud."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 2))


def _world_to_local(points, pose):
    """Express world ``points`` in the local frame of ``pose``."""
    x, y, yaw = pose
    delta = points - np.array([x, y])
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    return delta @ rot.T


def test_sequential_graph():
    """Five nodes with sequential edges build the expected graph."""
    graph = PoseGraph()
    for i in range(5):
        graph.add_node(Particle(float(i), 0.0, 0.0))
    for i in range(4):
        graph.add_edge(i, i + 1, _rel((i, 0.0, 0.0), (i + 1, 0.0, 0.0)))
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 4


def test_loop_closure_detected():
    """A node returning near node 0 triggers a loop-closure match."""
    graph = PoseGraph()
    poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
             (0.0, 1.0, 0.0), (0.05, 0.0, 0.0)]
    cloud = _cloud()
    scans = [cloud, _cloud(1), _cloud(2), _cloud(3), cloud]
    for pose in poses:
        graph.add_node(Particle(*pose))
    result = graph.detect_loop_closure(
        scans[4], scans, min_distance=0.5, icp_threshold=0.2, min_node_skip=2)
    assert result is not None
    candidate, _ = result
    assert candidate == 0


def test_optimize_triangle():
    """A triangular graph converges to the consistent solution."""
    gt = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, math.pi / 2.0)]
    graph = PoseGraph()
    graph.add_node(Particle(*gt[0]))
    graph.add_node(Particle(1.1, 0.05, 0.1))
    graph.add_node(Particle(0.9, 1.1, 1.4))
    graph.add_edge(0, 1, _rel(gt[0], gt[1]))
    graph.add_edge(1, 2, _rel(gt[1], gt[2]))
    graph.add_edge(0, 2, _rel(gt[0], gt[2]))
    optimized = graph.optimize('lm')
    for node, truth in zip(optimized[1:], gt[1:]):
        assert abs(node.x - truth[0]) < 0.02
        assert abs(node.y - truth[1]) < 0.02
        assert abs(node.yaw - truth[2]) < 0.02


def test_optimize_closes_loop():
    """Optimization with a loop-closure edge satisfies the closure constraint."""
    gt = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
          (1.0, 1.0, math.pi / 2.0), (0.0, 1.0, math.pi)]
    graph = PoseGraph()
    graph.add_node(Particle(*gt[0]))
    graph.add_node(Particle(1.1, 0.1, 0.05))
    graph.add_node(Particle(1.05, 1.1, 1.6))
    graph.add_node(Particle(-0.1, 1.05, 3.0))
    graph.add_edge(0, 1, _rel(gt[0], gt[1]))
    graph.add_edge(1, 2, _rel(gt[1], gt[2]))
    graph.add_edge(2, 3, _rel(gt[2], gt[3]))
    loop = _rel(gt[0], gt[3])
    graph.add_edge(0, 3, loop)
    optimized = graph.optimize('lm')

    mat0 = transform_matrix(optimized[0].x, optimized[0].y, optimized[0].yaw)
    mat3 = transform_matrix(optimized[3].x, optimized[3].y, optimized[3].yaw)
    err = np.linalg.inv(loop) @ np.linalg.inv(mat0) @ mat3
    dx, dy, dtheta = matrix_to_xytheta(err)
    assert math.hypot(dx, dy) < 0.05
    assert abs(dtheta) < 0.05


def test_reprojection_consistent():
    """Re-projected scans match the synthetic landmarks within tolerance."""
    landmarks = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [1.5, 1.5]])
    poses = [(0.0, 0.0, 0.0), (1.0, 0.5, math.pi / 4.0)]
    graph = PoseGraph()
    scan_history = []
    for pose in poses:
        graph.add_node(Particle(*pose))
        scan_history.append(_world_to_local(landmarks, pose))
    reproj = graph.reprojected_points(scan_history)
    expected = np.vstack([landmarks, landmarks])
    rmse = math.sqrt(np.mean((reproj - expected) ** 2))
    assert rmse < 0.1
