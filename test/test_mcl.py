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

"""Tests for the Monte Carlo Localizer."""

import math

import numpy as np

from slam_icp.mcl import MclConfig, MonteCarloLocalizer
from slam_icp.motion_model import MotionModelConfig
from slam_icp.particle import Particle
from slam_icp.sensor_model import LikelihoodFields


class FakeScan:
    """Minimal stand-in for ``sensor_msgs/LaserScan`` used in tests."""

    def __init__(self, ranges, angle_min, angle_increment,
                 range_min=0.1, range_max=10.0):
        """Store scan fields directly without requiring ROS."""
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max


def _box_grid(size=60, res=0.05):
    """Return a square grid bordered by occupied walls."""
    grid = np.zeros((size, size), dtype=np.int16)
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    return grid, res


def _simulate_scan(grid, res, origin_x, origin_y, pose,
                   n_beams=72, max_range=10.0):
    """Ray-march a scan against ``grid`` from ``pose`` (ground-truth scan)."""
    h, w = grid.shape
    angle_min = -math.pi
    inc = 2.0 * math.pi / n_beams
    ranges = []
    for k in range(n_beams):
        ang = pose.yaw + angle_min + k * inc
        r = 0.0
        hit = max_range
        while r < max_range:
            r += res * 0.5
            wx = pose.x + r * math.cos(ang)
            wy = pose.y + r * math.sin(ang)
            i = int((wx - origin_x) / res)
            j = int((wy - origin_y) / res)
            if i < 0 or i >= w or j < 0 or j >= h:
                hit = r
                break
            if grid[j, i] >= 50:
                hit = r
                break
        ranges.append(hit)
    return FakeScan(ranges, angle_min, inc)


def _make_mcl(adaptive, num_particles, rng):
    """Build an MCL with a likelihood-field sensor model over a box map."""
    grid, res = _box_grid()
    sensor = LikelihoodFields(std_dev=0.1, max_beams=0)
    sensor.compute_from_grid(grid, res, 0.0, 0.0)
    cfg = MclConfig(adaptive=adaptive, num_particles=num_particles,
                    min_particles=80, max_particles=250)
    mcl = MonteCarloLocalizer(cfg, MotionModelConfig(), sensor, rng)
    return mcl, grid, res


def test_initialize_count():
    """Initialization produces exactly N particles."""
    mcl, _, _ = _make_mcl(False, 300, np.random.default_rng(0))
    mcl.initialize(Particle(1.5, 1.5, 0.0, 1.0))
    assert len(mcl.get_particles()) == 300


def test_update_converges():
    """A fixed filter estimate stays within 0.2 m of the true pose."""
    rng = np.random.default_rng(1)
    mcl, grid, res = _make_mcl(False, 250, rng)
    true = Particle(1.5, 1.5, 0.0, 1.0)
    scan = _simulate_scan(grid, res, 0.0, 0.0, true)
    mcl.initialize(true, scale=0.15)
    estimate = mcl.update((0.0, 0.0, 0.0), scan)
    err = math.hypot(estimate.x - true.x, estimate.y - true.y)
    assert err < 0.2


def test_adaptive_shrinks():
    """A concentrated cloud drives the adaptive count toward the minimum."""
    rng = np.random.default_rng(2)
    mcl, grid, res = _make_mcl(True, 500, rng)
    true = Particle(1.5, 1.5, 0.0, 1.0)
    scan = _simulate_scan(grid, res, 0.0, 0.0, true)
    mcl.initialize(true, scale=0.02)
    mcl.update((0.0, 0.0, 0.0), scan)
    count = len(mcl.get_particles())
    assert count < 500
    assert count <= 250


def test_low_variance_resample_uniform():
    """Equal weights resample to a uniform, non-collapsed distribution."""
    rng = np.random.default_rng(3)
    mcl, _, _ = _make_mcl(False, 50, rng)
    n = 50
    particles = [Particle(float(i), 0.0, 0.0, 1.0 / n) for i in range(n)]
    resampled = mcl._low_variance_resample(particles, n)
    assert len(resampled) == n
    weights = [p.weight for p in resampled]
    assert all(abs(w - 1.0 / n) < 1e-12 for w in weights)
    unique_x = {round(p.x) for p in resampled}
    assert len(unique_x) == n
