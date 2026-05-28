"""
lidar_drive_test.py — drive 400 mm forward while recording lidar
=================================================================
Drives the robot straight ahead 400 mm (at ~80 mm/s), saves a lidar viz
frame every tick, and writes a final summary frame when done.

This is a pure orientation diagnostic: place objects in known positions
relative to the robot (e.g., a box 300 mm directly ahead), run the script,
and check that the viz shows them in the correct location.

HOW TO RUN
----------
    ROBOT_FSM_MODULE=robot.examples.lidar_drive_test ros2 run robot robot

Open the live viz:
    watch -n0.25 cp ros2_ws/runtime_output/lidar_viz.png /tmp/lv.png && feh /tmp/lv.png
Or just watch the file update in any image viewer that auto-refreshes.

CONTROLS
--------
  Enter  — start (resets odometry, then drives forward 400 mm)
  Ctrl-C — abort
"""

from __future__ import annotations

import math
import threading
import time

from robot.robot import FirmwareState, Robot, Unit
from robot.examples.lidar_viz import LidarViz
from robot.hardware_map import (
    DEFAULT_FSM_HZ,
    LED,
    LIDAR_FOV_DEG,
    LIDAR_MOUNT_THETA_DEG,
    LIDAR_MOUNT_X_MM,
    LIDAR_MOUNT_Y_MM,
    LIDAR_RANGE_MAX_MM,
    LIDAR_RANGE_MIN_MM,
    Motor,
)

# ---------------------------------------------------------------------------
# Hardware config — must match main.py
# ---------------------------------------------------------------------------

POSITION_UNIT            = Unit.MM
WHEEL_DIAMETER           = 74.0
WHEEL_BASE               = 321.0
INITIAL_THETA_DEG        = 90.0

LEFT_WHEEL_MOTOR         = Motor.DC_M2
LEFT_WHEEL_DIR_INVERTED  = False
RIGHT_WHEEL_MOTOR        = Motor.DC_M1
RIGHT_WHEEL_DIR_INVERTED = True

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

DRIVE_DISTANCE_MM  = 400.0   # how far to drive forward
DRIVE_SPEED_MM_S   = 80.0    # slow enough to see the lidar update
STATUS_INTERVAL_S  = 0.25    # print pose every N seconds

# For the viz: the "goal" is just the point 400 mm ahead of the start
GOAL_X_MM = 0.0
GOAL_Y_MM = DRIVE_DISTANCE_MM   # robot starts at theta=90° so forward = +Y world


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure(robot: Robot) -> None:
    robot.set_unit(POSITION_UNIT)
    robot.enable_lidar()
    robot.set_lidar_mount(
        x_mm=LIDAR_MOUNT_X_MM,
        y_mm=LIDAR_MOUNT_Y_MM,
        theta_deg=LIDAR_MOUNT_THETA_DEG,
    )
    robot.set_lidar_filter(
        range_min_mm=LIDAR_RANGE_MIN_MM,
        range_max_mm=LIDAR_RANGE_MAX_MM,
        fov_deg=LIDAR_FOV_DEG,
    )


def _set_odom_params(robot: Robot) -> bool:
    for attempt in range(5):
        ok = robot.set_odometry_parameters(
            wheel_diameter=WHEEL_DIAMETER,
            wheel_base=WHEEL_BASE,
            initial_theta_deg=INITIAL_THETA_DEG,
            left_motor_id=LEFT_WHEEL_MOTOR,
            left_motor_dir_inverted=LEFT_WHEEL_DIR_INVERTED,
            right_motor_id=RIGHT_WHEEL_MOTOR,
            right_motor_dir_inverted=RIGHT_WHEEL_DIR_INVERTED,
            timeout=2.0,
        )
        if ok:
            return True
        print(f"[lidar_test] odom not confirmed (attempt {attempt+1}), retrying…")
        time.sleep(0.2)
    return False


def _go_running(robot: Robot) -> None:
    for attempt in range(5):
        ok = robot.set_state(FirmwareState.RUNNING, timeout=10.0)
        if ok:
            print(f"[lidar_test] firmware → RUNNING (attempt {attempt+1})")
            return
        time.sleep(1.0)
    print("[lidar_test] WARNING: could not confirm RUNNING — continuing anyway")


def _reset_odom(robot: Robot) -> None:
    time.sleep(0.5)
    robot.reset_odometry()
    robot.wait_for_odometry_reset(timeout=3.0)
    x, y, theta = robot.get_pose()
    print(f"[lidar_test] odom reset  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(robot: Robot) -> None:
    try:
        _run(robot)
    finally:
        robot.stop()
        robot.set_led(LED.GREEN, 0)
        robot.set_led(LED.ORANGE, 255)
        print("[lidar_test] stopped")


def _run(robot: Robot) -> None:
    _configure(robot)

    if not _set_odom_params(robot):
        print("[lidar_test] FATAL: odom params never confirmed — aborting")
        return

    _go_running(robot)

    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 255)

    print()
    print("=" * 60)
    print("LIDAR DRIVE TEST — 400 mm straight forward")
    print()
    print("  Place an object ~400 mm directly in front of the robot.")
    print("  The viz should show it ABOVE the robot arrow (positive Y)")
    print("  and centred on the X axis.")
    print()
    print("  Open the viz:  ros2_ws/runtime_output/lidar_viz.png")
    print()
    print("  Press Enter to start, Ctrl-C to abort.")
    print("=" * 60)

    _enter_event = threading.Event()
    def _wait_enter():
        try:
            input()
        except EOFError:
            pass
        _enter_event.set()

    t = threading.Thread(target=_wait_enter, daemon=True)
    t.start()
    _enter_event.wait()

    _reset_odom(robot)

    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 255)

    viz = LidarViz(robot, goal=(GOAL_X_MM, GOAL_Y_MM))
    viz.start()
    print(f"[lidar_test] viz → ros2_ws/runtime_output/lidar_viz.png")

    print(f"[lidar_test] driving {DRIVE_DISTANCE_MM:.0f} mm forward at {DRIVE_SPEED_MM_S:.0f} mm/s …")
    handle = robot.move_forward(DRIVE_DISTANCE_MM, velocity=DRIVE_SPEED_MM_S, tolerance=20.0, blocking=False)

    period         = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick      = time.monotonic()
    last_status_at = time.monotonic()

    while True:
        now = time.monotonic()

        if now - last_status_at >= STATUS_INTERVAL_S:
            x, y, theta = robot.get_pose()
            raw_pts  = robot.get_obstacles()
            tracks   = robot.get_obstacle_tracks(include_unconfirmed=False)

            # Find the closest raw lidar point and its angle in robot frame
            closest_str = "none"
            if raw_pts:
                closest = min(raw_pts, key=lambda p: math.hypot(p[0], p[1]))
                d = math.hypot(closest[0], closest[1])
                a = math.degrees(math.atan2(closest[1], closest[0]))
                closest_str = f"({closest[0]:.0f},{closest[1]:.0f}) d={d:.0f}mm ang={a:.1f}°"

            print(f"  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°  "
                  f"raw={len(raw_pts)}  conf={len(tracks)}  "
                  f"closest_robot_frame={closest_str}")
            last_status_at = now

        if handle is not None and handle.is_finished():
            robot.stop()
            print("[lidar_test] DONE — 400 mm reached")
            x, y, theta = robot.get_pose()
            print(f"[lidar_test] final pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")
            print()
            print("CHECK: lidar dots in front of robot?")
            print("  → Should appear ABOVE the robot circle in the viz")
            print("  → Should be near X=0, Y=400 in world frame")
            # Let viz capture one more frame then stop
            time.sleep(1.0)
            viz.stop()
            return

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
