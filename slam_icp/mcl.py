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
Monte Carlo Localizer ported from ``mcl_cpp/src/monte_carlo_localizer.cpp``.

Adapted to operate over a map that is being built online rather than a static
map. Weights are accumulated in log-space and normalized with log-sum-exp for
numerical stability (PRD section 13.9), and resampling uses a low-variance
(stochastic universal) sampler (PRD section 4.6).
"""

from dataclasses import dataclass
import math
from typing import List, Optional

import numpy as np

from slam_icp.motion_model import (MotionModelConfig,
                                   sample_odom_delta_motion_model)
from slam_icp.particle import Particle
from slam_icp.sensor_model import LikelihoodFields

_ALMOST_ZERO = 1e-15


@dataclass
class MclConfig:
    """Particle-filter configuration (matches the ``mcl`` YAML block)."""

    adaptive: bool = True
    num_particles: int = 500
    epsilon: float = 0.05
    upper_quantile: float = 6.0
    min_particles: int = 80
    max_particles: int = 250
    resampling_threshold: float = 0.5
    kld_bin_size: float = 0.5
    kld_bin_yaw_deg: float = 10.0


class MonteCarloLocalizer:
    """Adaptive/fixed-size particle filter for 2D localization."""

    def __init__(self, config: MclConfig,
                 motion_cfg: Optional[MotionModelConfig] = None,
                 sensor_model: Optional[LikelihoodFields] = None,
                 rng: Optional[np.random.Generator] = None):
        """Wire up the filter with its config, motion and sensor models."""
        self.config = config
        self.motion_cfg = motion_cfg or MotionModelConfig()
        self.sensor_model = sensor_model
        self.rng = rng or np.random.default_rng()
        self.particles: List[Particle] = []
        self._pose_estimate = Particle(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, pose: Particle, n_particles: Optional[int] = None,
                   scale: float = 0.05) -> None:
        """Seed a Gaussian cloud of particles around ``pose``."""
        n = n_particles if n_particles is not None else self.config.num_particles
        if n <= 0:
            self.particles = []
            return
        weight = 1.0 / float(n)
        self.particles = []
        for _ in range(n - 1):
            x = self.rng.normal(pose.x, scale)
            y = self.rng.normal(pose.y, scale)
            yaw = self.rng.normal(pose.yaw, 0.01)
            self.particles.append(Particle(x, y, yaw, weight))
        self.particles.append(Particle(pose.x, pose.y, pose.yaw, weight))
        self._pose_estimate = Particle(pose.x, pose.y, pose.yaw, weight)

    # ------------------------------------------------------------------
    # Public update entry point
    # ------------------------------------------------------------------
    def update(self, odom_delta, scan, scan_points_base=None,
               occupancy_grid=None,
               resolution: Optional[float] = None, origin=None) -> Particle:
        """
        Advance the filter by one step and return the estimated pose.

        ``odom_delta`` is the relative motion ``(dx, dy, dyaw)`` expressed in
        the previous robot frame. When ``occupancy_grid`` plus ``resolution``
        and ``origin`` are supplied, the sensor distance field is recomputed
        first; otherwise the caller is expected to have refreshed it.
        """
        if (occupancy_grid is not None and resolution is not None
                and origin is not None and self.sensor_model is not None):
            self.sensor_model.compute_from_grid(
                occupancy_grid, resolution, origin[0], origin[1])

        if not self.particles:
            return self._pose_estimate

        if self.config.adaptive:
            self._update_adaptive(odom_delta, scan, scan_points_base)
        else:
            self._update_fixed(odom_delta, scan, scan_points_base)

        self._pose_estimate = self._weighted_mean(self.particles)
        return self._pose_estimate

    # ------------------------------------------------------------------
    # Fixed-size filter
    # ------------------------------------------------------------------
    def _update_fixed(self, odom_delta, scan, scan_points_base=None) -> None:
        predicted = []
        log_weights = np.empty(len(self.particles), dtype=float)
        for i, particle in enumerate(self.particles):
            pred = sample_odom_delta_motion_model(
                particle, odom_delta, self.motion_cfg, self.rng)
            predicted.append(pred)
            log_weights[i] = self._log_weight(
                scan, pred, scan_points_base=scan_points_base)

        weights = self._softmax(log_weights)
        for pred, w in zip(predicted, weights):
            pred.weight = float(w)
        self.particles = predicted

        n = len(self.particles)
        if self._should_resample(self.particles,
                                 self.config.resampling_threshold * n):
            self.particles = self._low_variance_resample(self.particles, n)

    # ------------------------------------------------------------------
    # Adaptive (KLD) filter
    # ------------------------------------------------------------------
    def _update_adaptive(self, odom_delta, scan, scan_points_base=None) -> None:
        cum = self._cumulative_weights(self.particles)

        new_particles: List[Particle] = []
        log_weights: List[float] = []
        bins = set()
        n_bins = 0
        required = float(self.config.min_particles)
        bin_xy = max(self.config.kld_bin_size, 1e-6)
        bin_yaw = max(self.config.kld_bin_yaw_deg, 1e-6)

        while len(new_particles) < int(required):
            sampled = self._sample_from_cumulative(self.particles, cum)
            pred = sample_odom_delta_motion_model(
                sampled, odom_delta, self.motion_cfg, self.rng)
            new_particles.append(pred)
            log_weights.append(self._log_weight(
                scan, pred, scan_points_base=scan_points_base))

            key = (
                int(math.floor(pred.x / bin_xy)),
                int(math.floor(pred.y / bin_xy)),
                int(math.floor(math.degrees(pred.yaw) / bin_yaw)),
            )
            if key not in bins:
                bins.add(key)
                n_bins += 1

            if n_bins > 1:
                required = self._kld_required(
                    n_bins, self.config.epsilon, self.config.upper_quantile)
            required = max(required, float(self.config.min_particles))
            required = min(required, float(self.config.max_particles))

        weights = self._softmax(np.asarray(log_weights, dtype=float))
        for pred, w in zip(new_particles, weights):
            pred.weight = float(w)
        self.particles = new_particles

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------
    def _log_weight(self, scan, particle: Particle,
                    scan_points_base=None) -> float:
        if self.sensor_model is None:
            return 0.0
        return self.sensor_model.log_update(
            scan, particle, scan_points_base=scan_points_base)

    @staticmethod
    def _softmax(log_weights: np.ndarray) -> np.ndarray:
        """Normalize log-weights into a probability vector (log-sum-exp)."""
        n = log_weights.shape[0]
        if n == 0:
            return log_weights
        if not np.all(np.isfinite(log_weights)):
            return np.full(n, 1.0 / n)
        max_log = log_weights.max()
        weights = np.exp(log_weights - max_log)
        total = weights.sum()
        if total < _ALMOST_ZERO:
            return np.full(n, 1.0 / n)
        return weights / total

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    @staticmethod
    def _should_resample(particles: List[Particle], threshold: float) -> bool:
        sum_sq = sum(p.weight * p.weight for p in particles)
        if sum_sq <= 0.0:
            return False
        return (1.0 / sum_sq) < threshold

    def _low_variance_resample(self, particles: List[Particle],
                               n: int) -> List[Particle]:
        """Stochastic universal (low-variance) resampler."""
        weights = np.array([p.weight for p in particles], dtype=float)
        total = weights.sum()
        if total < _ALMOST_ZERO:
            weights = np.full(len(particles), 1.0 / len(particles))
        else:
            weights = weights / total
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0

        start = self.rng.uniform(0.0, 1.0 / n)
        positions = start + np.arange(n) / n
        new_weight = 1.0 / n

        resampled: List[Particle] = []
        idx = 0
        for pos in positions:
            while pos > cumsum[idx]:
                idx += 1
            src = particles[idx]
            resampled.append(Particle(src.x, src.y, src.yaw, new_weight))
        return resampled

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _cumulative_weights(particles: List[Particle]) -> np.ndarray:
        return np.cumsum([p.weight for p in particles])

    def _sample_from_cumulative(self, particles: List[Particle],
                                cum: np.ndarray) -> Particle:
        r = self.rng.uniform(1e-6, 1.0)
        m = 0
        last = len(cum) - 1
        while m < last and cum[m] < r:
            m += 1
        return particles[m]

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------
    @staticmethod
    def _weighted_mean(particles: List[Particle]) -> Particle:
        if not particles:
            return Particle(0.0, 0.0, 0.0, 0.0)
        x = sum(p.x * p.weight for p in particles)
        y = sum(p.y * p.weight for p in particles)
        # Circular mean for yaw to handle angle wrapping correctly.
        sin_sum = sum(math.sin(p.yaw) * p.weight for p in particles)
        cos_sum = sum(math.cos(p.yaw) * p.weight for p in particles)
        yaw = math.atan2(sin_sum, cos_sum)
        return Particle(x, y, yaw, 1.0)

    def best_particle(self) -> Particle:
        """Return the highest-weight particle."""
        return max(self.particles, key=lambda p: p.weight)

    def get_particles(self) -> List[Particle]:
        """Return the current particle list."""
        return self.particles

    @property
    def pose_estimate(self) -> Particle:
        """Return the latest weighted-mean pose estimate."""
        return self._pose_estimate

    @staticmethod
    def _kld_required(k: int, epsilon: float, upper_quantile: float) -> float:
        """Fox (2003) KLD sample-size bound via the Wilson-Hilferty transform."""
        k_minus_one = float(k - 1)
        x = (1.0 - 2.0 / (9.0 * k_minus_one)
             + math.sqrt(2.0 / (9.0 * k_minus_one)) * upper_quantile)
        return math.ceil(k_minus_one / (2.0 * epsilon) * x * x * x)
