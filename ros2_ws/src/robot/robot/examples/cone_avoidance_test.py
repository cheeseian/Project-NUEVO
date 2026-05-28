"""
cone_avoidance_test.py — LAPF corridor run from the actual track entrance
=========================================================================
Place the robot at the cone-corridor entrance (physical position 1525, 350
on the full map).  Press Enter in the terminal to start.  The robot drives
3000 mm straight ahead using LAPF cone avoidance with the same parameters
used in the full competition run.

HOW TO RUN (no need to overwrite main.py)
-----------------------------------------
    ROBOT_FSM_MODULE=robot.examples.cone_avoidance_test ros2 run robot robot

Or via run_robot.sh by setting the env var before launching the container command.

CONTROLS
--------
  Enter — start the run (resets odometry then begins LAPF)
  Ctrl-C — stop everything
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
# LAPF parameters — mirrors main.py LAPF_* constants
# ---------------------------------------------------------------------------

# Goal is 3000 mm forward (+Y) from the odometry-reset origin.
# Robot is physically placed at the corridor entrance; after reset pose=(0,0,90°).
GOAL_X_MM            = 0.0
GOAL_Y_MM            = 3000.0

LAPF_VELOCITY_MM_S   = 120.0
LAPF_TOLERANCE_MM    = 100.0
LAPF_MAX_ANGULAR     = 1.2
LAPF_LEASH_MM        = 250.0
LAPF_HALF_ANGLE_DEG  = 85.0
LAPF_REPULSION_MM    = 430.0
LAPF_INFLATION_MM    = 75.0
LAPF_TARGET_SPD_MM_S = 200.0
LAPF_REPULSION_GAIN  = 200.0
LAPF_ATTRACTION_GAIN = 3.0
LAPF_EMA_ALPHA       = 0.35

STATUS_INTERVAL_S    = 0.5

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
    """Send odom params while firmware is IDLE; retry up to 5×."""
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
            p = robot.get_odometry_parameters()
            print(f"[cone_test] odom confirmed (attempt {attempt+1}): "
                  f"L=M{p['left_motor_number']} inv={p['left_motor_dir_inverted']}  "
                  f"R=M{p['right_motor_number']} inv={p['right_motor_dir_inverted']}")
            return True
        print(f"[cone_test] odom not confirmed (attempt {attempt+1}), retrying…")
        time.sleep(0.2)
    return False


def _go_running(robot: Robot) -> None:
    for attempt in range(5):
        ok = robot.set_state(FirmwareState.RUNNING, timeout=10.0)
        if ok:
            print(f"[cone_test] firmware → RUNNING (attempt {attempt+1})")
            return
        print(f"[cone_test] set_state RUNNING failed (attempt {attempt+1}), retrying…")
        time.sleep(1.0)
    print("[cone_test] WARNING: could not confirm RUNNING — continuing anyway")


def _reset_odom(robot: Robot) -> None:
    time.sleep(0.5)
    robot.reset_odometry()
    robot.wait_for_odometry_reset(timeout=3.0)
    x, y, theta = robot.get_pose()
    print(f"[cone_test] odom reset  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")
    if abs(theta - INITIAL_THETA_DEG) > 5.0:
        print(f"[cone_test] WARNING: theta={theta:.1f}° expected {INITIAL_THETA_DEG}° — resetting again")
        robot.reset_odometry()
        robot.wait_for_odometry_reset(timeout=3.0)
        x, y, theta = robot.get_pose()
        print(f"[cone_test] re-reset  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")


def _start_lapf(robot: Robot):
    return robot.lapf_to_goal(
        GOAL_X_MM,
        GOAL_Y_MM,
        velocity=LAPF_VELOCITY_MM_S,
        tolerance=LAPF_TOLERANCE_MM,
        leash_length_mm=LAPF_LEASH_MM,
        repulsion_range_mm=LAPF_REPULSION_MM,
        target_speed_mm_s=LAPF_TARGET_SPD_MM_S,
        max_angular_rad_s=LAPF_MAX_ANGULAR,
        repulsion_gain=LAPF_REPULSION_GAIN,
        attraction_gain=LAPF_ATTRACTION_GAIN,
        force_ema_alpha=LAPF_EMA_ALPHA,
        inflation_margin_mm=LAPF_INFLATION_MM,
        leash_half_angle_deg=LAPF_HALF_ANGLE_DEG,
        blocking=False,
    )


def _print_status(robot: Robot) -> None:
    x, y, theta = robot.get_pose()
    vt          = robot.get_virtual_target()
    confirmed   = robot.get_obstacle_tracks(include_unconfirmed=False)
    raw_pts     = robot.get_obstacles()
    vt_str      = f"vt=({vt[0]:.0f},{vt[1]:.0f})" if vt is not None else "vt=none"
    remaining   = math.hypot(GOAL_X_MM - x, GOAL_Y_MM - y)
    if confirmed:
        nearest = min(confirmed, key=lambda o: math.hypot(o["x"] - x, o["y"] - y))
        nd = math.hypot(nearest["x"] - x, nearest["y"] - y)
        near_str = f"nearest=({nearest['x']:.0f},{nearest['y']:.0f}) d={nd:.0f}mm"
    else:
        near_str = "nearest=none"
    print(f"  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°  rem={remaining:.0f}mm  "
          f"{vt_str}  raw={len(raw_pts)}  conf={len(confirmed)}  {near_str}")


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
        print("[cone_test] stopped")


def _run(robot: Robot) -> None:
    # ── 1. Configure while firmware is IDLE ──────────────────────────────────
    _configure(robot)

    if not _set_odom_params(robot):
        print("[cone_test] FATAL: odom params never confirmed — aborting")
        return

    _go_running(robot)

    # ── 2. Wait for Enter ─────────────────────────────────────────────────────
    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 255)

    print()
    print("=" * 60)
    print("CONE AVOIDANCE TEST — corridor entrance → 3000 mm ahead")
    print(f"  Goal: ({GOAL_X_MM:.0f}, {GOAL_Y_MM:.0f}) mm  speed: {LAPF_VELOCITY_MM_S:.0f} mm/s")
    print(f"  repulsion_gain={LAPF_REPULSION_GAIN}  attraction_gain={LAPF_ATTRACTION_GAIN}")
    print(f"  half_angle={LAPF_HALF_ANGLE_DEG}°  leash={LAPF_LEASH_MM:.0f}mm")
    print()
    print("  Place robot at corridor entrance, then press Enter to start.")
    print("  Ctrl-C to abort at any time.")
    print("=" * 60)

    # Read Enter in a background thread so ROS2 keeps spinning
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

    # ── 3. Reset odom and go ──────────────────────────────────────────────────
    _reset_odom(robot)

    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 255)

    viz = LidarViz(robot, goal=(GOAL_X_MM, GOAL_Y_MM))
    viz.start()
    print("[cone_test] visualiser → ros2_ws/runtime_output/lidar_viz.png (open and watch it)")

    print("[cone_test] starting LAPF…")
    handle = _start_lapf(robot)

    period         = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick      = time.monotonic()
    last_status_at = time.monotonic()

    while True:
        now = time.monotonic()

        if now - last_status_at >= STATUS_INTERVAL_S:
            _print_status(robot)
            last_status_at = now

        if handle is not None and handle.is_finished():
            robot.stop()
            viz.stop()
            print("[cone_test] DONE — goal reached")
            _print_status(robot)
            return

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
