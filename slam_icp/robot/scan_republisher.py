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

"""Republish LaserScan with a consistent frame_id for RViz/TF.

Note: rotating the URDF or changing ``frame_id`` here only affects RViz.
``slam_icp_node`` must apply the laser->base extrinsic via TF when building
the map (``tf.use_scan_extrinsic`` in YAML).
"""

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanRepublisher(Node):
    """Copy ``/scan`` to ``/scan_fixed`` with a consistent ``frame_id``."""

    def __init__(self):
        super().__init__('scan_republisher')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_fixed')
        self.declare_parameter('scan_frame', 'laser')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.scan_frame = str(self.get_parameter('scan_frame').value)

        self.pub = self.create_publisher(LaserScan, output_topic, 10)
        self.create_subscription(LaserScan, input_topic, self._scan_callback, 10)

        self.get_logger().info(
            f'republishing {input_topic} -> {output_topic} '
            f'(frame_id={self.scan_frame})')

    def _scan_callback(self, msg):
        fixed_msg = copy.deepcopy(msg)
        fixed_msg.header.stamp = msg.header.stamp
        fixed_msg.header.frame_id = self.scan_frame
        self.pub.publish(fixed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
