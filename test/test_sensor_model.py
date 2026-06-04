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

"""Tests for the likelihood-field sensor model."""

import math

import numpy as np

from slam_icp.particle import Particle
from slam_icp.sensor_model import LikelihoodFields


class FakeScan:
    """Minimal stand-in for ``sensor_msgs/LaserScan`` used in tests."""

    def __init__(self, ranges, angle_min=0.0, angle_increment=math.radians(1.0),
                 range_min=0.1, range_max=10.0):
        """Store scan fields directly without requiring ROS."""
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max


def _wall_grid(width=40, height=40, wall_col=20):
    """Return a grid with a vertical occupied wall and free elsewhere."""
    grid = np.zeros((height, width), dtype=np.int16)
    grid[:, wall_col] = 100
    return grid


def test_correct_pose_high_weight():
    """A beam hitting the wall from the right pose yields a high weight."""
    model = LikelihoodFields(std_dev=0.05, max_beams=0)
    model.compute_from_grid(_wall_grid(), 0.05, 0.0, 0.0)
    scan = FakeScan([0.5])
    weight = model.update(scan, Particle(0.5, 0.5, 0.0))
    assert weight > 0.5


def test_incorrect_pose_low_weight():
    """A beam pointing away from the wall yields a low weight."""
    model = LikelihoodFields(std_dev=0.05, max_beams=0)
    model.compute_from_grid(_wall_grid(), 0.05, 0.0, 0.0)
    scan = FakeScan([0.5])
    weight = model.update(scan, Particle(0.5, 0.5, math.pi))
    assert weight < 0.1


def test_empty_map_equal_weights():
    """With no obstacles, every pose receives the same weight."""
    model = LikelihoodFields(std_dev=0.05, max_beams=0)
    model.compute_from_grid(np.zeros((40, 40), dtype=np.int16), 0.05, 0.0, 0.0)
    scan = FakeScan([0.5])
    w_a = model.update(scan, Particle(0.5, 0.5, 0.0))
    w_b = model.update(scan, Particle(1.2, 0.3, 1.0))
    assert w_a == w_b == 1.0


def test_none_scan_returns_one():
    """A ``None`` scan returns weight 1.0, matching mcl_cpp."""
    model = LikelihoodFields(std_dev=0.05)
    model.compute_from_grid(_wall_grid(), 0.05, 0.0, 0.0)
    assert model.update(None, Particle(0.5, 0.5, 0.0)) == 1.0


def test_log_space_normalization():
    """Log-weights over 360 beams stay finite and normalize to one."""
    model = LikelihoodFields(std_dev=0.05, max_beams=0)
    model.compute_from_grid(_wall_grid(), 0.05, 0.0, 0.0)
    scan = FakeScan([0.5] * 360)
    poses = [Particle(0.5, 0.5, 0.0), Particle(0.55, 0.5, 0.0),
             Particle(0.4, 0.6, 0.2), Particle(0.5, 0.5, math.pi)]
    log_weights = np.array([model.log_update(scan, p) for p in poses])
    assert np.all(np.isfinite(log_weights))
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    assert np.all(np.isfinite(weights))
    assert abs(weights.sum() - 1.0) < 1e-9
