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

"""Differential-drive odometry from wheel encoder angular velocities."""

import math

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WheelOdometryNode(Node):
    """Integrate ``/VelocityEncL`` and ``/VelocityEncR`` into ``/odom`` + TF."""

    def __init__(self):
        super().__init__('wheel_odometry')

        self.last_time = self.get_clock().now()

        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_base', 0.19)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('left_encoder_topic', '/VelocityEncL')
        self.declare_parameter('right_encoder_topic', '/VelocityEncR')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('update_rate_hz', 20.0)
        self.declare_parameter('linear_sign', 1.0)
        self.declare_parameter('angular_sign', 1.0)
        self.declare_parameter('swap_wheels', False)

        self.r = float(self.get_parameter('wheel_radius').value)
        self.L = float(self.get_parameter('wheel_base').value)
        self.linear_sign = float(self.get_parameter('linear_sign').value)
        self.angular_sign = float(self.get_parameter('angular_sign').value)
        self.swap_wheels = bool(self.get_parameter('swap_wheels').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        left_topic = str(self.get_parameter('left_encoder_topic').value)
        right_topic = str(self.get_parameter('right_encoder_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        update_rate = float(self.get_parameter('update_rate_hz').value)

        self.w_l = 0.0
        self.w_r = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.create_subscription(Float32, left_topic, self._left_callback, qos)
        self.create_subscription(Float32, right_topic, self._right_callback, qos)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        period = 1.0 / max(update_rate, 1.0)
        self.create_timer(period, self._update)

        self.get_logger().info(
            f'wheel odometry: encoders -> {odom_topic} + TF '
            f'{self.odom_frame}->{self.base_frame} '
            f'(linear_sign={self.linear_sign}, '
            f'angular_sign={self.angular_sign}, '
            f'swap_wheels={self.swap_wheels})')

    def _left_callback(self, msg):
        self.w_l = float(msg.data)

    def _right_callback(self, msg):
        self.w_r = float(msg.data)

    def _update(self):
        now_clock = self.get_clock().now()
        dt = (now_clock - self.last_time).nanoseconds / 1e9
        self.last_time = now_clock

        if dt <= 0.0 or dt > 0.2:
            return

        w_l = self.w_r if self.swap_wheels else self.w_l
        w_r = self.w_l if self.swap_wheels else self.w_r

        v = self.linear_sign * self.r * (w_r + w_l) / 2.0
        w = self.angular_sign * self.r * (w_r - w_l) / self.L

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = normalize_angle(self.yaw + w * dt)

        now = now_clock.to_msg()
        q = yaw_to_quaternion(self.yaw)

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = q
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = w
        self.odom_pub.publish(odom_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = q
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryNode()
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
