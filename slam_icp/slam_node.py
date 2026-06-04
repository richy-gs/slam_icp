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

"""ROS 2 SLAM node fusing ICP scan matching, MCL, and pose-graph loop closure."""

import math
import threading

from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
import rclpy
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from slam_icp.icp import icp
from slam_icp.map_builder import MapBuilder
from slam_icp.mcl import MclConfig, MonteCarloLocalizer
from slam_icp.motion_model import MotionModelConfig
from slam_icp.particle import Particle
from slam_icp.pose_graph import PoseGraph
from slam_icp.scan_utils import (filter_valid_points, laserscan_to_base_points,
                                 laserscan_to_points)
from slam_icp.sensor_model import LikelihoodFields
from slam_icp.util import (angle_diff, matrix_to_xytheta, quaternion_from_yaw,
                           transform_matrix, yaw_from_quaternion)

# Bootstrap states (PRD section 13.5).
WAITING_TF = 'WAITING_TF'
WAITING_FIRST_SCAN = 'WAITING_FIRST_SCAN'
WARMUP = 'WARMUP'
RUNNING = 'RUNNING'


class SlamIcpNode(Node):
    """Particle-filter SLAM node publishing /map, /slam_pose and TF map->odom."""

    def __init__(self):
        """Declare parameters, build modules, and wire up pubs/subs/TF."""
        super().__init__('slam_icp_node')
        self._declare_parameters()
        self._load_parameters()

        # Algorithm modules.
        origin = (self._map_origin_x, self._map_origin_y)
        self._map_builder = MapBuilder(
            self._map_resolution, self._map_width, self._map_height, origin)
        self._sensor_model = LikelihoodFields(
            std_dev=self._sensor_std_dev, max_beams=self._sensor_max_beams,
            max_range=self._scan_max_range)
        self._rng = np.random.default_rng()
        self._mcl = MonteCarloLocalizer(
            self._mcl_cfg, self._motion_cfg, self._sensor_model, self._rng)
        self._pose_graph = PoseGraph()
        self._scan_history = []

        # State and shared data (guarded by _lock).
        self._lock = threading.Lock()
        self._state = WAITING_TF
        self._latest_odom = None
        self._prev_odom = None
        self._slam_pose = None
        self._last_keyframe_pose = None
        self._last_keyframe_scan = None
        self._last_keyframe_id = -1
        self._warmup_count = 0
        self._keyframe_count = 0
        self._map_to_odom = (0.0, 0.0, 0.0)
        self._optimizing = False

        # TF.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Callback groups: scan is mutually exclusive, everything else fast.
        self._scan_group = MutuallyExclusiveCallbackGroup()
        self._fast_group = ReentrantCallbackGroup()

        scan_qos = (qos_profile_sensor_data if self._scan_best_effort
                    else QoSProfile(depth=5))
        odom_qos = QoSProfile(depth=10,
                              reliability=ReliabilityPolicy.RELIABLE)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, scan_qos,
            callback_group=self._scan_group)
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, odom_qos,
            callback_group=self._fast_group)

        self._map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)
        self._particle_pub = self.create_publisher(
            PoseArray, '/particlecloud', 10)
        self._slam_path_pub = self.create_publisher(Path, '/slam_path', 10)
        self._odom_path_pub = self.create_publisher(Path, '/odom_path', 10)

        self._slam_path = Path()
        self._odom_path = Path()

        period = 1.0 / max(self._publish_rate_hz, 0.1)
        self._tf_timer = self.create_timer(
            period, self._publish_tf, callback_group=self._fast_group)

        self.get_logger().info('slam_icp_node started; waiting for TF/scan.')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        self.declare_parameters('', [
            ('publish_rate_hz', 5.0),
            ('use_icp_for_initial_pose', True),
            ('icp.max_iterations', 50),
            ('icp.tolerance', 1.0e-4),
            ('icp.max_correspondence_dist', 0.5),
            ('scan.max_range', 10.0),
            ('scan.min_range', 0.1),
            ('scan.max_beams', 360),
            ('motion_model.alpha1', 0.001),
            ('motion_model.alpha2', 0.001),
            ('motion_model.alpha3', 0.010),
            ('motion_model.alpha4', 0.001),
            ('sensor_model.std_dev', 0.05),
            ('sensor_model.max_beams', 30),
            ('sensor_model.model', 'likelihood_field'),
            ('mcl.adaptive', True),
            ('mcl.num_particles', 500),
            ('mcl.epsilon', 0.05),
            ('mcl.upper_quantile', 6.0),
            ('mcl.min_particles', 80),
            ('mcl.max_particles', 250),
            ('mcl.resampling_threshold', 0.5),
            ('map.resolution', 0.05),
            ('map.width', 400),
            ('map.height', 400),
            ('map.origin_x', -10.0),
            ('map.origin_y', -10.0),
            ('pose_graph.loop_closure_min_distance', 0.5),
            ('pose_graph.loop_closure_icp_threshold', 0.1),
            ('pose_graph.loop_closure_min_node_skip', 20),
            ('pose_graph.optimization_every_n_nodes', 10),
            ('bootstrap.tf_wait_timeout_sec', 30.0),
            ('bootstrap.warmup_scans', 3),
            ('keyframe.min_distance_m', 0.30),
            ('keyframe.min_yaw_rad', 0.20),
            ('icp_runtime.use_odom_seed', True),
            ('icp_runtime.run_scan_to_map', True),
            ('icp_runtime.min_correspondence_ratio', 0.30),
            ('icp_runtime.reject_rmse_threshold', 0.50),
            ('sensor_model_runtime.use_log_likelihood', True),
            ('sensor_model_runtime.dt_recompute_every_n_keyframes', 5),
            ('pose_graph_runtime.optimize_in_thread', True),
            ('pose_graph_runtime.max_history_keyframes', 2000),
            ('tf.base_frame', 'base_footprint'),
            ('tf.odom_frame', 'odom'),
            ('tf.map_frame', 'map'),
            ('tf.scan_frame', 'base_scan'),
            ('tf.use_scan_extrinsic', True),
            ('tf.scan_yaw_offset', 0.0),
            ('tf.scan_extrinsic_fallback_x', 0.053),
            ('tf.scan_extrinsic_fallback_y', 0.0),
            ('tf.scan_extrinsic_fallback_yaw', 0.0),
            ('tf.lookup_timeout_sec', 0.10),
            ('qos.scan_best_effort', True),
            ('qos.map_transient_local', True),
        ])

    def _get(self, name):
        return self.get_parameter(name).value

    def _load_parameters(self):
        self._publish_rate_hz = float(self._get('publish_rate_hz'))
        self._use_icp = bool(self._get('use_icp_for_initial_pose'))

        self._icp_max_iter = int(self._get('icp.max_iterations'))
        self._icp_tol = float(self._get('icp.tolerance'))
        self._icp_max_corr = float(self._get('icp.max_correspondence_dist'))

        self._scan_max_range = float(self._get('scan.max_range'))
        self._scan_min_range = float(self._get('scan.min_range'))

        self._motion_cfg = MotionModelConfig(
            alpha1=float(self._get('motion_model.alpha1')),
            alpha2=float(self._get('motion_model.alpha2')),
            alpha3=float(self._get('motion_model.alpha3')),
            alpha4=float(self._get('motion_model.alpha4')))

        self._sensor_std_dev = float(self._get('sensor_model.std_dev'))
        self._sensor_max_beams = int(self._get('sensor_model.max_beams'))

        self._mcl_cfg = MclConfig(
            adaptive=bool(self._get('mcl.adaptive')),
            num_particles=int(self._get('mcl.num_particles')),
            epsilon=float(self._get('mcl.epsilon')),
            upper_quantile=float(self._get('mcl.upper_quantile')),
            min_particles=int(self._get('mcl.min_particles')),
            max_particles=int(self._get('mcl.max_particles')),
            resampling_threshold=float(self._get('mcl.resampling_threshold')),
            kld_bin_size=float(self._get('map.resolution')) * 10.0)

        self._map_resolution = float(self._get('map.resolution'))
        self._map_width = int(self._get('map.width'))
        self._map_height = int(self._get('map.height'))
        self._map_origin_x = float(self._get('map.origin_x'))
        self._map_origin_y = float(self._get('map.origin_y'))

        self._lc_min_dist = float(self._get('pose_graph.loop_closure_min_distance'))
        self._lc_icp_thresh = float(
            self._get('pose_graph.loop_closure_icp_threshold'))
        self._lc_min_skip = int(
            self._get('pose_graph.loop_closure_min_node_skip'))
        self._opt_every_n = int(
            self._get('pose_graph.optimization_every_n_nodes'))

        self._warmup_scans = int(self._get('bootstrap.warmup_scans'))
        self._kf_min_dist = float(self._get('keyframe.min_distance_m'))
        self._kf_min_yaw = float(self._get('keyframe.min_yaw_rad'))

        self._use_odom_seed = bool(self._get('icp_runtime.use_odom_seed'))
        self._icp_min_ratio = float(
            self._get('icp_runtime.min_correspondence_ratio'))
        self._icp_reject_rmse = float(
            self._get('icp_runtime.reject_rmse_threshold'))

        self._dt_recompute_n = int(
            self._get('sensor_model_runtime.dt_recompute_every_n_keyframes'))
        self._optimize_in_thread = bool(
            self._get('pose_graph_runtime.optimize_in_thread'))
        self._max_history = int(
            self._get('pose_graph_runtime.max_history_keyframes'))

        self._base_frame = str(self._get('tf.base_frame'))
        self._odom_frame = str(self._get('tf.odom_frame'))
        self._map_frame = str(self._get('tf.map_frame'))
        self._scan_frame = str(self._get('tf.scan_frame'))
        self._use_scan_extrinsic = bool(self._get('tf.use_scan_extrinsic'))
        self._scan_yaw_offset = float(self._get('tf.scan_yaw_offset'))
        self._scan_extrinsic_fallback = (
            float(self._get('tf.scan_extrinsic_fallback_x')),
            float(self._get('tf.scan_extrinsic_fallback_y')),
            float(self._get('tf.scan_extrinsic_fallback_yaw')),
        )
        self._lookup_timeout = float(self._get('tf.lookup_timeout_sec'))
        self._warned_scan_extrinsic = False

        self._scan_best_effort = bool(self._get('qos.scan_best_effort'))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        try:
            pose = msg.pose.pose
            yaw = yaw_from_quaternion(pose.orientation.x, pose.orientation.y,
                                      pose.orientation.z, pose.orientation.w)
            with self._lock:
                self._latest_odom = (pose.position.x, pose.position.y, yaw)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'odom_callback failed: {exc}',
                                    throttle_duration_sec=2.0)

    def _scan_callback(self, msg: LaserScan):
        try:
            self._process_scan(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'scan_callback failed: {exc}',
                                    throttle_duration_sec=2.0)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def _scan_to_base_points(self, msg: LaserScan):
        """Convert scan ranges to ``base_frame`` points using TF extrinsics."""
        if self._use_scan_extrinsic:
            points = laserscan_to_base_points(
                msg, self._scan_max_range, self._tf_buffer,
                self._base_frame, self._scan_frame, self._lookup_timeout,
                min_range=self._scan_min_range,
                use_extrinsic=True,
                fallback=self._scan_extrinsic_fallback,
                yaw_offset=self._scan_yaw_offset)
            if (not self._warned_scan_extrinsic
                    and msg.header.frame_id != self._base_frame):
                self.get_logger().info(
                    f'scan extrinsic: {msg.header.frame_id} -> '
                    f'{self._base_frame} (TF + fallback)')
                self._warned_scan_extrinsic = True
            return points

        return filter_valid_points(
            laserscan_to_points(msg, self._scan_max_range),
            self._scan_min_range)

    def _process_scan(self, msg: LaserScan):
        if not msg.ranges:
            return
        ranges = np.asarray(msg.ranges, dtype=float)
        if not np.any(np.isfinite(ranges)):
            self.get_logger().warn('scan has no finite ranges',
                                   throttle_duration_sec=2.0)
            return

        points = self._scan_to_base_points(msg)
        if points.shape[0] < 5:
            self.get_logger().warn('scan too sparse after filtering',
                                   throttle_duration_sec=2.0)
            return

        with self._lock:
            odom = self._latest_odom

        if odom is None:
            self._publish_identity_tf()
            return

        if self._state == WAITING_TF:
            self._state = WAITING_FIRST_SCAN

        if self._slam_pose is None:
            self._bootstrap_first_scan(points, odom)
            return

        odom_delta = self._odom_delta(self._prev_odom, odom)
        self._prev_odom = odom

        if self._state == WARMUP:
            self._slam_pose = self._integrate(self._slam_pose, odom_delta)
            self._map_builder.update(points, self._slam_pose)
            self._warmup_count += 1
            if self._warmup_count >= self._warmup_scans:
                self._recompute_sensor_field()
                self._state = RUNNING
                self.get_logger().info('warmup complete; SLAM running.')
            self._publish_all(odom)
            return

        # RUNNING: MCL update over the current map.
        self._mcl.update(odom_delta, msg, scan_points_base=points)
        self._slam_pose = self._mcl.pose_estimate

        self._maybe_add_keyframe(points)
        self._publish_all(odom)

    def _bootstrap_first_scan(self, points, odom):
        self._slam_pose = Particle(0.0, 0.0, 0.0, 1.0)
        self._prev_odom = odom
        self._mcl.initialize(self._slam_pose)
        self._map_builder.update(points, self._slam_pose)
        self._last_keyframe_pose = Particle(0.0, 0.0, 0.0, 1.0)
        self._last_keyframe_scan = points
        self._last_keyframe_id = self._pose_graph.add_node(self._slam_pose)
        self._scan_history.append(points)
        self._keyframe_count = 1
        self._warmup_count = 0
        self._state = WARMUP
        self.get_logger().info('first scan received; map seeded at origin.')
        self._publish_all(odom)

    def _maybe_add_keyframe(self, points):
        dx = self._slam_pose.x - self._last_keyframe_pose.x
        dy = self._slam_pose.y - self._last_keyframe_pose.y
        dist = math.hypot(dx, dy)
        dyaw = abs(angle_diff(self._slam_pose.yaw,
                              self._last_keyframe_pose.yaw))
        if dist < self._kf_min_dist and dyaw < self._kf_min_yaw:
            return

        # Refine the keyframe-to-keyframe motion with ICP when possible.
        refined_pose = self._slam_pose
        if self._use_icp and self._last_keyframe_scan is not None:
            seed = self._relative_transform(self._last_keyframe_pose,
                                            self._slam_pose)
            transform, rmse, success = icp(
                points, self._last_keyframe_scan,
                max_iterations=self._icp_max_iter, tolerance=self._icp_tol,
                max_correspondence_dist=self._icp_max_corr,
                init_transform=seed,
                min_correspondence_ratio=self._icp_min_ratio)
            if success and rmse < self._icp_reject_rmse:
                kf_mat = transform_matrix(self._last_keyframe_pose.x,
                                          self._last_keyframe_pose.y,
                                          self._last_keyframe_pose.yaw)
                world = kf_mat @ transform
                rx, ry, ryaw = matrix_to_xytheta(world)
                refined_pose = Particle(rx, ry, ryaw, 1.0)
                self._slam_pose = refined_pose

        node_id = self._pose_graph.add_node(refined_pose)
        edge_t = self._relative_transform(self._last_keyframe_pose,
                                          refined_pose)
        self._pose_graph.add_edge(self._last_keyframe_id, node_id, edge_t)
        self._scan_history.append(points)
        self._map_builder.update(points, refined_pose)

        self._last_keyframe_pose = refined_pose
        self._last_keyframe_scan = points
        self._last_keyframe_id = node_id
        self._keyframe_count += 1

        if self._keyframe_count % self._dt_recompute_n == 0:
            self._recompute_sensor_field()

        self._try_loop_closure(points)

    def _try_loop_closure(self, points):
        if self._optimizing:
            return
        result = self._pose_graph.detect_loop_closure(
            points, self._scan_history,
            min_distance=self._lc_min_dist,
            icp_threshold=self._lc_icp_thresh,
            min_node_skip=self._lc_min_skip,
            max_correspondence_dist=self._icp_max_corr)
        if result is None:
            return
        candidate_id, transform = result
        self._pose_graph.add_edge(candidate_id, self._last_keyframe_id,
                                  transform)
        self.get_logger().info(
            f'Loop closure detected at node {candidate_id}.')
        if self._optimize_in_thread:
            thread = threading.Thread(target=self._run_optimization,
                                      daemon=True)
            thread.start()
        else:
            self._run_optimization()

    def _run_optimization(self):
        self._optimizing = True
        try:
            optimized = self._pose_graph.optimize(method='lm')
            self._remap(optimized)
            self.get_logger().info('pose graph optimization complete.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'optimization failed: {exc}',
                                    throttle_duration_sec=2.0)
        finally:
            self._optimizing = False

    def _remap(self, optimized):
        origin = (self._map_origin_x, self._map_origin_y)
        new_map = MapBuilder(self._map_resolution, self._map_width,
                             self._map_height, origin)
        with self._lock:
            history = list(self._scan_history)
        for k, pose in enumerate(optimized):
            if k >= len(history):
                break
            new_map.update(history[k], pose)
        with self._lock:
            self._map_builder = new_map
            if optimized:
                last = optimized[-1]
                self._slam_pose = Particle(last.x, last.y, last.yaw, 1.0)
                self._last_keyframe_pose = self._slam_pose
        self._recompute_sensor_field()

    def _recompute_sensor_field(self):
        grid = self._map_builder.to_numpy()
        self._sensor_model.compute_from_grid(
            grid, self._map_resolution, self._map_origin_x, self._map_origin_y)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _odom_delta(prev, curr):
        """Relative motion ``prev->curr`` expressed in ``prev`` robot frame."""
        if prev is None:
            return (0.0, 0.0, 0.0)
        prev_mat = transform_matrix(prev[0], prev[1], prev[2])
        curr_mat = transform_matrix(curr[0], curr[1], curr[2])
        rel = np.linalg.inv(prev_mat) @ curr_mat
        return matrix_to_xytheta(rel)

    @staticmethod
    def _integrate(pose, delta):
        """Compose ``pose`` with a relative ``delta`` in the pose frame."""
        pose_mat = transform_matrix(pose.x, pose.y, pose.yaw)
        delta_mat = transform_matrix(delta[0], delta[1], delta[2])
        out = pose_mat @ delta_mat
        x, y, yaw = matrix_to_xytheta(out)
        return Particle(x, y, yaw, 1.0)

    @staticmethod
    def _relative_transform(pose_a, pose_b):
        """Return ``T`` such that ``pose_a . T == pose_b`` (3x3 matrix)."""
        mat_a = transform_matrix(pose_a.x, pose_a.y, pose_a.yaw)
        mat_b = transform_matrix(pose_b.x, pose_b.y, pose_b.yaw)
        return np.linalg.inv(mat_a) @ mat_b

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_all(self, odom):
        self._publish_map()
        self._publish_pose()
        self._publish_particles()
        self._publish_paths(odom)
        self._compute_map_to_odom(odom)
        self._publish_tf()

    def _publish_map(self):
        msg = self._map_builder.get_map()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        self._map_pub.publish(msg)

    def _publish_pose(self):
        if self._slam_pose is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.pose = self._slam_pose.to_pose_msg()
        self._pose_pub.publish(msg)

    def _publish_particles(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.poses = [p.to_pose_msg() for p in self._mcl.get_particles()]
        self._particle_pub.publish(msg)

    def _publish_paths(self, odom):
        stamp = self.get_clock().now().to_msg()
        if self._slam_pose is not None:
            self._slam_path.header.stamp = stamp
            self._slam_path.header.frame_id = self._map_frame
            pose = PoseStamped()
            pose.header = self._slam_path.header
            pose.pose = self._slam_pose.to_pose_msg()
            self._slam_path.poses.append(pose)
            self._slam_path_pub.publish(self._slam_path)

        self._odom_path.header.stamp = stamp
        self._odom_path.header.frame_id = self._odom_frame
        odom_pose = PoseStamped()
        odom_pose.header = self._odom_path.header
        odom_pose.pose = Particle(odom[0], odom[1], odom[2]).to_pose_msg()
        self._odom_path.poses.append(odom_pose)
        self._odom_path_pub.publish(self._odom_path)

    def _compute_map_to_odom(self, odom):
        """Compute ``map->odom`` from the SLAM pose and the odom->base TF."""
        if self._slam_pose is None:
            return
        map_base = transform_matrix(
            self._slam_pose.x, self._slam_pose.y, self._slam_pose.yaw)
        odom_base = transform_matrix(odom[0], odom[1], odom[2])
        map_odom = map_base @ np.linalg.inv(odom_base)
        self._map_to_odom = matrix_to_xytheta(map_odom)

    def _publish_tf(self):
        x, y, yaw = self._map_to_odom
        self._send_transform(x, y, yaw)

    def _publish_identity_tf(self):
        self._send_transform(0.0, 0.0, 0.0)

    def _send_transform(self, x, y, yaw):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._odom_frame
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    """Spin the SLAM node under a multi-threaded executor."""
    rclpy.init(args=args)
    node = SlamIcpNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
