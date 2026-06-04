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

"""Odometry motion model ported from ``mcl_cpp/src/motion_model.cpp``."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from slam_icp.particle import Particle
from slam_icp.util import angle_diff

_MOVED_TOO_CLOSE = 0.01


@dataclass
class MotionModelConfig:
    """Thrun odometry-model noise parameters ``alpha1..alpha4``."""

    alpha1: float = 0.001
    alpha2: float = 0.001
    alpha3: float = 0.010
    alpha4: float = 0.001


def _xytheta(pose) -> Tuple[float, float, float]:
    """Coerce a Particle, tuple, or ``(x, y, yaw)`` sequence into a tuple."""
    if hasattr(pose, 'yaw'):
        return float(pose.x), float(pose.y), float(pose.yaw)
    return float(pose[0]), float(pose[1]), float(pose[2])


def calculate_pose_delta(curr, prev) -> Tuple[float, float, float]:
    """Decompose motion from ``prev`` to ``curr`` into ``(rot1, trans, rot2)``."""
    x, y, theta = _xytheta(prev)
    x_prime, y_prime, theta_prime = _xytheta(curr)

    delta_translation = math.sqrt((x - x_prime) ** 2 + (y - y_prime) ** 2)

    delta_rotation1 = 0.0
    if delta_translation > _MOVED_TOO_CLOSE:
        delta_rotation1 = angle_diff(
            math.atan2(y_prime - y, x_prime - x), theta)
    delta = angle_diff(theta_prime, theta)
    delta_rotation2 = angle_diff(delta, delta_rotation1)

    return delta_rotation1, delta_translation, delta_rotation2


def _sample_normal(std_dev: float, rng: Optional[np.random.Generator]) -> float:
    """Sample from ``N(0, std_dev)``; returns 0 for non-positive ``std_dev``."""
    if std_dev <= 0.0:
        return 0.0
    if rng is None:
        return float(np.random.normal(0.0, std_dev))
    return float(rng.normal(0.0, std_dev))


def sample_odom_motion_model(
    particle: Particle,
    prev_odom,
    curr_odom,
    cfg: MotionModelConfig,
    rng: Optional[np.random.Generator] = None,
) -> Particle:
    """Propagate ``particle`` by the noisy odometry delta ``prev_odom→curr_odom``."""
    prev = _xytheta(prev_odom)
    curr = _xytheta(curr_odom)
    if prev == curr:
        return Particle(particle.x, particle.y, particle.yaw, particle.weight)

    d1, dt, d2 = calculate_pose_delta(curr, prev)

    a1, a2, a3, a4 = cfg.alpha1, cfg.alpha2, cfg.alpha3, cfg.alpha4
    std_dev_d1 = math.sqrt(a1 * d1 * d1 + a2 * dt * dt)
    std_dev_dt = math.sqrt(a3 * dt * dt + a4 * d1 * d1 + a4 * d2 * d2)
    std_dev_d2 = math.sqrt(a1 * d2 * d2 + a2 * dt * dt)

    noise_d1 = _sample_normal(std_dev_d1, rng)
    noise_dt = _sample_normal(std_dev_dt, rng)
    noise_d2 = _sample_normal(std_dev_d2, rng)

    t_d1 = angle_diff(d1, noise_d1)
    t_dt = dt + noise_dt
    t_d2 = angle_diff(d2, noise_d2)

    curr_x, curr_y, curr_yaw = particle.x, particle.y, particle.yaw
    x = curr_x + t_dt * math.cos(curr_yaw + t_d1)
    y = curr_y + t_dt * math.sin(curr_yaw + t_d1)
    yaw = curr_yaw + t_d1 + t_d2

    return Particle(x=x, y=y, yaw=yaw, weight=particle.weight)
