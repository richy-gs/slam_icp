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

"""Tests for the occupancy-grid map builder."""

import numpy as np

from slam_icp.map_builder import MapBuilder
from slam_icp.particle import Particle
from slam_icp.util import map_to_world, world_to_map

_RES = 0.05
_ORIGIN = (-5.0, -5.0)


def _builder():
    """Return a fresh 200x200 map builder at the standard origin."""
    return MapBuilder(_RES, 200, 200, _ORIGIN)


def test_front_point_occupied():
    """A scan point 1 m ahead marks the corresponding cell occupied."""
    builder = _builder()
    builder.update(np.array([[1.0, 0.0]]), Particle(0.0, 0.0, 0.0))
    grid = builder.to_numpy()
    col, row = world_to_map(1.0, 0.0, _ORIGIN[0], _ORIGIN[1], _RES)
    assert grid[row, col] > 50


def test_raycast_marks_free():
    """Cells between the robot and the hit point become free."""
    builder = _builder()
    builder.update(np.array([[1.0, 0.0]]), Particle(0.0, 0.0, 0.0))
    grid = builder.to_numpy()
    col, row = world_to_map(0.5, 0.0, _ORIGIN[0], _ORIGIN[1], _RES)
    assert grid[row, col] != -1
    assert grid[row, col] < 50


def test_repeated_updates_accumulate():
    """Repeated hits raise the occupancy probability (log-odds)."""
    single = _builder()
    single.update(np.array([[1.0, 0.0]]), Particle(0.0, 0.0, 0.0))
    col, row = world_to_map(1.0, 0.0, _ORIGIN[0], _ORIGIN[1], _RES)
    v_single = single.to_numpy()[row, col]

    repeated = _builder()
    for _ in range(5):
        repeated.update(np.array([[1.0, 0.0]]), Particle(0.0, 0.0, 0.0))
    v_repeated = repeated.to_numpy()[row, col]
    assert v_repeated > v_single


def test_to_numpy_value_range():
    """The numpy grid has the right shape and only valid occupancy values."""
    builder = _builder()
    builder.update(np.array([[1.0, 0.0], [0.0, 1.0]]), Particle(0.0, 0.0, 0.0))
    grid = builder.to_numpy()
    assert grid.shape == (200, 200)
    valid = (grid == -1) | ((grid >= 0) & (grid <= 100))
    assert valid.all()


def test_world_map_roundtrip():
    """``world_to_map`` and ``map_to_world`` are inverses on valid cells."""
    for i in range(0, 200, 17):
        for j in range(0, 200, 23):
            x, y = map_to_world(i, j, _ORIGIN[0], _ORIGIN[1], _RES)
            ri, rj = world_to_map(x, y, _ORIGIN[0], _ORIGIN[1], _RES)
            assert (ri, rj) == (i, j)


def test_out_of_bounds_no_exception():
    """Points far outside the grid are ignored without raising."""
    builder = _builder()
    builder.update(np.array([[1000.0, 1000.0]]), Particle(0.0, 0.0, 0.0))
    grid = builder.to_numpy()
    assert grid.shape == (200, 200)
