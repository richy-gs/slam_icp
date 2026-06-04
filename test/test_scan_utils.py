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

"""Tests for LaserScan conversion and point-cloud transforms."""

import math

import numpy as np

from slam_icp.scan_utils import (filter_valid_points, laserscan_to_points,
                                 transform_points)


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


def test_360_rays():
    """A 360-beam scan produces at most 360 valid points."""
    ranges = [1.0] * 360
    points = laserscan_to_points(FakeScan(ranges))
    assert points.shape[1] == 2
    assert points.shape[0] <= 360
    assert points.shape[0] == 360


def test_invalid_rays_filtered():
    """Non-finite ranges are dropped from the output."""
    ranges = [1.0, float('inf'), float('nan'), 2.0]
    points = laserscan_to_points(FakeScan(ranges))
    assert points.shape[0] == 2


def test_filter_valid_points_min_range():
    """Points closer than ``min_range`` and non-finite ones are removed."""
    pts = np.array([[0.05, 0.0], [1.0, 0.0], [np.inf, 0.0]])
    out = filter_valid_points(pts, min_range=0.1)
    assert out.shape[0] == 1
    np.testing.assert_allclose(out[0], [1.0, 0.0])


def test_transform_translation():
    """A pose of (1, 0, 0) translates the points along x."""
    pts = np.array([[1.0, 0.0]])
    out = transform_points(pts, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(out[0], [2.0, 0.0], atol=1e-9)


def test_transform_rotation():
    """A 90 degree pose rotates (1, 0) onto (0, 1)."""
    pts = np.array([[1.0, 0.0]])
    out = transform_points(pts, (0.0, 0.0, math.pi / 2.0))
    np.testing.assert_allclose(out[0], [0.0, 1.0], atol=1e-9)
