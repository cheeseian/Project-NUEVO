"""
Robot GPS node.

Bridges global ArUco localizer detections from the Jetson Nano into the
robot's local ROS2 domain.

The Jetson publishes on /global_gps/tag_detections across the full network
(ROS_LOCALHOST_ONLY=0).  The rest of the robot stack runs with
ROS_LOCALHOST_ONLY=1 and cannot see that topic.  This node runs with
ROS_LOCALHOST_ONLY=0, receives the Jetson's detections, and re-publishes
them on /tag_detections so that localhost-only nodes (the robot node, etc.)
can subscribe normally.

IMPORTANT — startup:
    This node must be launched with ROS_LOCALHOST_ONLY=0 explicitly,
    separate from the main bridge/robot stack.  See the sensors package
    README or ros2_ws/README.md for the exact command.

Topic subscribed  (global network):
    /global_gps/tag_detections  (bridge_interfaces/msg/TagDetectionArray)

Topic published  (visible locally on this machine):
    /tag_detections  (bridge_interfaces/msg/TagDetectionArray)
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from bridge_interfaces.msg import TagDetectionArray


_RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)


class RobotGpsNode(Node):
    """Re-publishes global GPS detections into the local ROS domain."""

    def __init__(self) -> None:
        super().__init__("robot_gps")

        self._pub = self.create_publisher(
            TagDetectionArray,
            "/tag_detections",
            _RELIABLE_QOS,
        )

        self.create_subscription(
            TagDetectionArray,
            "/global_gps/tag_detections",
            self._on_detections,
            _RELIABLE_QOS,
        )

        self.get_logger().info(
            "Robot GPS node started. "
            "Bridging /global_gps/tag_detections → /tag_detections"
        )

    def _on_detections(self, msg: TagDetectionArray) -> None:
        self._pub.publish(msg)
        self.get_logger().debug(
            f"Forwarded {len(msg.detections)} detection(s): "
            f"{[d.tag_id for d in msg.detections]}"
        )


def main() -> None:
    rclpy.init()
    node = RobotGpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
