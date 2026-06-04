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

"""Tests for the odometry motion model."""

import math

import numpy as np

from slam_icp.motion_model import (
    MotionModelConfig,
    sample_odom_delta_motion_model,
    sample_odom_motion_model,
)
from slam_icp.particle import Particle


def _sample_many(particle, prev, curr, cfg, n=2000, seed=0):
    """Draw ``n`` motion samples and return their (x, y, yaw) arrays."""
    rng = np.random.default_rng(seed)
    xs, ys, yaws = [], [], []
    for _ in range(n):
        out = sample_odom_motion_model(particle, prev, curr, cfg, rng)
        xs.append(out.x)
        ys.append(out.y)
        yaws.append(out.yaw)
    return np.array(xs), np.array(ys), np.array(yaws)


def test_no_motion():
    """Equal odometry poses leave the particle unchanged."""
    cfg = MotionModelConfig()
    particle = Particle(0.3, -0.2, 0.5, 1.0)
    rng = np.random.default_rng(0)
    out = sample_odom_motion_model(
        particle, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), cfg, rng)
    assert out.x == particle.x
    assert out.y == particle.y
    assert out.yaw == particle.yaw


def test_forward_one_meter():
    """Driving one meter forward centers the cloud near (1, 0)."""
    cfg = MotionModelConfig()
    particle = Particle(0.0, 0.0, 0.0, 1.0)
    xs, ys, _ = _sample_many(
        particle, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), cfg)
    assert abs(xs.mean() - 1.0) < 0.05
    assert abs(ys.mean()) < 0.05
    assert xs.std() < 0.2


def test_rotate_90():
    """Rotating 90 degrees centers the yaw distribution at pi/2."""
    cfg = MotionModelConfig()
    particle = Particle(0.0, 0.0, 0.0, 1.0)
    _, _, yaws = _sample_many(
        particle, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2.0), cfg)
    assert abs(yaws.mean() - math.pi / 2.0) < 0.05
    assert yaws.std() < 0.2


def test_delta_forward_in_particle_frame():
    """Delta (1,0,0) advances along the particle heading, not global +X."""
    cfg = MotionModelConfig(alpha1=0.0, alpha2=0.0, alpha3=0.0, alpha4=0.0)
    particle = Particle(0.0, 0.0, math.pi / 2.0, 1.0)
    rng = np.random.default_rng(0)
    out = sample_odom_delta_motion_model(
        particle, (1.0, 0.0, 0.0), cfg, rng)
    assert abs(out.x) < 1e-9
    assert abs(out.y - 1.0) < 1e-9
    assert abs(out.yaw - math.pi / 2.0) < 1e-9


def test_zero_noise_deterministic():
    """With all alphas zero the model is deterministic."""
    cfg = MotionModelConfig(alpha1=0.0, alpha2=0.0, alpha3=0.0, alpha4=0.0)
    particle = Particle(0.0, 0.0, 0.0, 1.0)
    rng = np.random.default_rng(0)
    out = sample_odom_motion_model(
        particle, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), cfg, rng)
    assert abs(out.x - 1.0) < 1e-9
    assert abs(out.y) < 1e-9
    assert abs(out.yaw) < 1e-9
