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

"""Tests for the ICP scan-matching module."""

import math

import numpy as np

from slam_icp.icp import apply_transform, icp
from slam_icp.util import transform_matrix


def _cloud(seed=0, n=80):
    """Return a reproducible asymmetric 2D point cloud."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 2))


def test_identity():
    """Identical clouds yield an identity transform and zero RMSE."""
    cloud = _cloud()
    transform, rmse, success = icp(cloud, cloud, max_correspondence_dist=5.0)
    assert success
    assert rmse < 1e-6
    np.testing.assert_allclose(transform, np.eye(3), atol=1e-6)


def test_rotation():
    """A 15 degree rotation is recovered within one degree."""
    cloud = _cloud()
    theta = math.radians(15.0)
    target = apply_transform(cloud, transform_matrix(0.0, 0.0, theta))
    transform, rmse, success = icp(cloud, target, max_correspondence_dist=5.0)
    recovered = math.atan2(transform[1, 0], transform[0, 0])
    assert success
    assert abs(recovered - theta) < math.radians(1.0)


def test_translation():
    """A pure translation is recovered to centimeter accuracy."""
    cloud = _cloud()
    target = cloud + np.array([1.0, 0.5])
    transform, rmse, success = icp(cloud, target, max_correspondence_dist=5.0)
    assert success
    assert abs(transform[0, 2] - 1.0) < 0.01
    assert abs(transform[1, 2] - 0.5) < 0.01


def test_empty_cloud():
    """An empty source returns identity and an infinite RMSE without raising."""
    transform, rmse, success = icp(np.empty((0, 2)), _cloud())
    assert not success
    assert math.isinf(rmse)
    np.testing.assert_allclose(transform, np.eye(3))


def test_outliers():
    """ICP converges despite 10 percent of spurious far points."""
    cloud = _cloud(n=100)
    theta = math.radians(10.0)
    target = apply_transform(cloud, transform_matrix(0.0, 0.0, theta))
    rng = np.random.default_rng(5)
    n_out = 10
    outliers = rng.uniform(8.0, 12.0, size=(n_out, 2))
    noisy_source = np.vstack((cloud, outliers))
    transform, rmse, success = icp(
        noisy_source, target, max_correspondence_dist=0.5)
    recovered = math.atan2(transform[1, 0], transform[0, 0])
    assert success
    assert abs(recovered - theta) < math.radians(2.0)


def test_seed_helps_large_rotation():
    """An odometry seed lets ICP recover a large 30 degree rotation."""
    cloud = _cloud(n=120)
    theta = math.radians(30.0)
    target = apply_transform(cloud, transform_matrix(0.0, 0.0, theta))
    seed = transform_matrix(0.0, 0.0, theta)
    _, rmse_seeded, success_seeded = icp(
        cloud, target, max_correspondence_dist=5.0, init_transform=seed)
    _, rmse_noseed, _ = icp(cloud, target, max_correspondence_dist=5.0)
    assert success_seeded
    assert rmse_seeded < 1e-3
    assert rmse_seeded <= rmse_noseed + 1e-6


def test_low_correspondence_fails():
    """Disjoint clouds give too few correspondences and report failure."""
    source = _cloud(seed=1, n=60)
    target = _cloud(seed=2, n=60) + np.array([50.0, 50.0])
    _, _, success = icp(source, target, max_correspondence_dist=0.5,
                        min_correspondence_ratio=0.3)
    assert not success
