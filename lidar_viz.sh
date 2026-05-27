#!/usr/bin/env bash
# Stream LiDAR point cloud from inside Docker and display it with matplotlib on the host.
# Usage: ./lidar_viz.sh
set -e

CONTAINER="docker-ros2_runtime-1"
ROS_SETUP="source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== LiDAR Visualizer ==="
echo "Streaming /scan from ${CONTAINER} → matplotlib"
echo "Close the window or Ctrl+C to stop."
echo ""

docker exec -i "$CONTAINER" bash -c \
    "$ROS_SETUP && python3 /ros2_ws/src/scan_dump.py" \
    | python3 "$SCRIPT_DIR/lidar_viz_host.py"
