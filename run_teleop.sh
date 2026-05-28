#!/usr/bin/env bash
# Launch the robot in arrow-key teleoperation mode.
# Requires an interactive terminal (ssh -t or a local shell).
set -e

CONTAINER="docker-ros2_runtime-1"
ROS_SETUP="source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash"
BUILD_PKG_DIR="/ros2_ws/build/robot/robot"
SRC_PKG_DIR="/ros2_ws/src/robot/robot"

_exec()    { docker exec         "$CONTAINER" bash -c "$*"; }
_exec_it() { docker exec -it     "$CONTAINER" bash -c "$*"; }

_cleanup() {
    echo ""
    echo "[run_teleop] stopping robot..."
    docker exec "$CONTAINER" bash -c \
        "kill -2 \$(ps aux | grep '[r]os2_ws/install/robot/lib/robot/robot' | awk '{print \$2}') 2>/dev/null; true"
    sleep 2
    docker exec "$CONTAINER" bash -c \
        "kill -9 \$(ps aux | grep '[r]os2_ws/install/robot/lib/robot/robot' | awk '{print \$2}') 2>/dev/null; true"
    docker exec "$CONTAINER" bash -c \
        "$ROS_SETUP && \
         ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
             '{motor_number: 1, target_ticks: 0}' > /dev/null 2>&1; \
         ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
             '{motor_number: 2, target_ticks: 0}' > /dev/null 2>&1; true"
    echo "[run_teleop] robot stopped."
    exit 0
}
trap _cleanup INT TERM

echo "=== run_teleop.sh ==="

# ── 1. Kill stale robot nodes ─────────────────────────────────────────────────
echo "[1/5] killing stale robot nodes..."
_exec "kill -9 \$(ps aux | grep '[r]os2_ws/install/robot/lib/robot/robot' | awk '{print \$2}') 2>/dev/null; true"

# ── 2. Sync teleop.py and robot_node.py into the build directory ──────────────
# (src and build may already be symlinked — suppress same-file errors)
echo "[2/5] syncing teleop files to build dir..."
_exec "cp ${SRC_PKG_DIR}/teleop.py     ${BUILD_PKG_DIR}/teleop.py     2>/dev/null || true"
_exec "cp ${SRC_PKG_DIR}/robot_node.py ${BUILD_PKG_DIR}/robot_node.py 2>/dev/null || true"

# ── 3. Reset firmware → IDLE, then → RUNNING ─────────────────────────────────
echo "[3/5] resetting firmware..."
_exec "$ROS_SETUP && ros2 topic pub --once /sys_cmd bridge_interfaces/msg/SysCommand \
    '{command: 3}' > /dev/null 2>&1; true"
sleep 1
_exec "$ROS_SETUP && ros2 topic pub --once /sys_cmd bridge_interfaces/msg/SysCommand \
    '{command: 1}' > /dev/null 2>&1; true"
sleep 1

# ── 4. Prime motor direction ──────────────────────────────────────────────────
echo "[4/5] priming motor direction..."
_exec "$ROS_SETUP && \
    ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
        '{motor_number: 1, target_ticks: 10}' > /dev/null 2>&1; \
    ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
        '{motor_number: 2, target_ticks: 10}' > /dev/null 2>&1; true"
sleep 0.3
_exec "$ROS_SETUP && \
    ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
        '{motor_number: 1, target_ticks: 0}' > /dev/null 2>&1; \
    ros2 topic pub --once /dc_set_velocity bridge_interfaces/msg/DCSetVelocity \
        '{motor_number: 2, target_ticks: 0}' > /dev/null 2>&1; true"
sleep 0.5

# ── 5. Launch teleop (interactive TTY required for arrow-key input) ───────────
echo "[5/5] starting teleop..."
_exec_it "$ROS_SETUP && ROBOT_FSM_MODULE=robot.teleop ros2 run robot robot"
