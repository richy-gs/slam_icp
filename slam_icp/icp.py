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

"""Iterative Closest Point (ICP) for rigid 2D scan matching, pure NumPy/SciPy."""

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous 2D transform to an ``(N, 2)`` point array."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] == 0:
        return points.copy()
    rot = transform[:2, :2]
    trans = transform[:2, 2]
    return points @ rot.T + trans


def _best_fit_transform(source: np.ndarray,
                        target: np.ndarray) -> np.ndarray:
    """Closed-form least-squares rigid transform mapping ``source`` to ``target``."""
    centroid_src = source.mean(axis=0)
    centroid_tgt = target.mean(axis=0)
    src_centered = source - centroid_src
    tgt_centered = target - centroid_tgt

    cov = src_centered.T @ tgt_centered
    u_mat, _, vt_mat = np.linalg.svd(cov)
    rot = vt_mat.T @ u_mat.T
    # Guard against reflections (det(R) == -1).
    if np.linalg.det(rot) < 0.0:
        vt_mat = vt_mat.copy()
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T
    trans = centroid_tgt - rot @ centroid_src

    transform = np.eye(3, dtype=float)
    transform[:2, :2] = rot
    transform[:2, 2] = trans
    return transform


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    max_correspondence_dist: float = 0.5,
    init_transform: Optional[np.ndarray] = None,
    min_correspondence_ratio: float = 0.3,
) -> Tuple[np.ndarray, float, bool]:
    """
    Align ``source`` onto ``target`` with point-to-point ICP.

    Returns a tuple ``(T, rmse, success)`` where ``T`` is the 3x3 homogeneous
    transform that maps ``source`` into ``target``'s frame, ``rmse`` is the
    final residual error and ``success`` reports whether the alignment is
    trustworthy (enough correspondences and a finite residual).
    """
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    target = np.asarray(target, dtype=float).reshape(-1, 2)

    if init_transform is None:
        transform = np.eye(3, dtype=float)
    else:
        transform = np.asarray(init_transform, dtype=float).copy()

    if source.shape[0] == 0 or target.shape[0] == 0:
        return transform, float('inf'), False

    tree = cKDTree(target)
    src = apply_transform(source, transform)
    prev_rmse = float('inf')
    rmse = float('inf')
    ratio = 0.0

    for _ in range(max_iterations):
        distances, indices = tree.query(src)
        valid = distances <= max_correspondence_dist
        n_valid = int(np.count_nonzero(valid))
        ratio = n_valid / float(source.shape[0])

        if n_valid < 3:
            # Too few correspondences to estimate a stable transform.
            break

        matched_src = src[valid]
        matched_tgt = target[indices[valid]]

        step = _best_fit_transform(matched_src, matched_tgt)
        transform = step @ transform
        src = apply_transform(source, transform)

        residual = tree.query(src)[0]
        residual = residual[residual <= max_correspondence_dist]
        if residual.size == 0:
            rmse = float('inf')
            break
        rmse = float(math.sqrt(np.mean(residual ** 2)))

        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

    success = (
        math.isfinite(rmse)
        and ratio >= min_correspondence_ratio
    )
    return transform, rmse, success
