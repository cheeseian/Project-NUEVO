#!/usr/bin/env python3
"""Stream /scan as space-separated x y integers (mm) to stdout, one line per scan."""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

RANGE_MIN_M = 0.15
RANGE_MAX_M = 6.0


class ScanDump(Node):
    def __init__(self):
        super().__init__('scan_dump')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos)

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float32)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(r), dtype=np.float32)
        mask = (r >= RANGE_MIN_M) & (r <= RANGE_MAX_M) & np.isfinite(r)
        r, a = r[mask], angles[mask]
        x = ( r * np.cos(a) * 1000.0).astype(np.int32)
        y = (-r * np.sin(a) * 1000.0).astype(np.int32)  # negate to match nav code (upside-down mount)
        line = ' '.join(map(str, np.column_stack((x, y)).flatten()))
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


rclpy.init()
node = ScanDump()
rclpy.spin(node)
