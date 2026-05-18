"""
cone_avoidance_test.py — drive 2 m forward, avoid two cones
============================================================
Drives straight ahead 2000 mm using LAPF (Leashed Artificial Potential Field)
obstacle avoidance. Place two cones anywhere in the 2 m corridor; the robot
will steer around them and reach the goal.

HOW TO RUN
----------
    cp examples/cone_avoidance_test.py main.py
    ros2 run robot robot

CONTROLS
--------
  BTN_1 — start / restart run (resets odometry each time)
  BTN_2 — cancel active motion and return to idle

TUNING
------
Edit the constants below. Key ones for cone avoidance:
  REPULSION_RANGE_MM   — robot starts steering when a cone is within this radius
  INFLATION_MARGIN_MM  — extra buffer added around each detected obstacle cluster
  LEASH_LENGTH_MM      — how far ahead the virtual target is projected
"""

from __future__ import annotations

import time

from robot.hardware_map import (
    DEFAULT_FSM_HZ,
    INITIAL_THETA_DEG,
    LED,
    LIDAR_FOV_DEG,
    LIDAR_MOUNT_THETA_DEG,
    LIDAR_MOUNT_X_MM,
    LIDAR_MOUNT_Y_MM,
    LIDAR_RANGE_MAX_MM,
    LIDAR_RANGE_MIN_MM,
    POSITION_UNIT,
    WHEEL_BASE,
    WHEEL_DIAMETER,
)
from robot.hardware_map import Motor

LEFT_WHEEL_MOTOR         = Motor.DC_M2
LEFT_WHEEL_DIR_INVERTED  = False
RIGHT_WHEEL_MOTOR        = Motor.DC_M1
RIGHT_WHEEL_DIR_INVERTED = True
from robot.robot import FirmwareState, Robot


# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------

# Goal: 2 metres straight ahead in the odometry world frame.
# With INITIAL_THETA_DEG=90° the robot faces world +Y, so forward = +Y.
GOAL_X_MM = 0.0
GOAL_Y_MM = 4000.0

VELOCITY_MM_S    = 130.0   # cruise speed
TOLERANCE_MM     = 100.0   # accept goal when within this radius
MAX_ANGULAR_RAD_S = 1.2    # cap on turning rate

# ---------------------------------------------------------------------------
# LAPF tuning — adjust these for your cone size and corridor width
# ---------------------------------------------------------------------------

LEASH_LENGTH_MM      = 220.0   # virtual-target lookahead distance
LEASH_HALF_ANGLE_DEG = 50.0    # cone of directions the virtual target can sit in
REPULSION_RANGE_MM   = 500.0   # start avoiding when obstacle centre is within this
INFLATION_MARGIN_MM  = 260.0   # extra bubble added around each obstacle cluster
TARGET_SPEED_MM_S    = 200.0   # speed used inside the LAPF force calculation
REPULSION_GAIN       = 600.0
ATTRACTION_GAIN      = 1.0
FORCE_EMA_ALPHA      = 0.35    # smoothing on force vector (0 = no smoothing)

STATUS_PRINT_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_robot(robot: Robot) -> None:
    robot.set_unit(POSITION_UNIT)
    robot.set_odometry_parameters(
        wheel_diameter=WHEEL_DIAMETER,
        wheel_base=WHEEL_BASE,
        initial_theta_deg=INITIAL_THETA_DEG,
        left_motor_id=LEFT_WHEEL_MOTOR,
        left_motor_dir_inverted=LEFT_WHEEL_DIR_INVERTED,
        right_motor_id=RIGHT_WHEEL_MOTOR,
        right_motor_dir_inverted=RIGHT_WHEEL_DIR_INVERTED,
    )
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


def start_robot(robot: Robot) -> None:
    current = robot.get_state()
    if current in (FirmwareState.ESTOP, FirmwareState.ERROR):
        robot.reset_estop()
    robot.set_state(FirmwareState.RUNNING)


def reset_pose(robot: Robot) -> None:
    for attempt in range(5):
        robot.reset_odometry()
        if not robot.wait_for_odometry_reset(timeout=2.0):
            robot.wait_for_pose_update(timeout=0.5)
        x, y, _ = robot.get_odometry_pose()
        if abs(x) < 50.0 and abs(y) < 50.0:
            return
        print(f"[warn] odometry not zeroed (got {x:.0f}, {y:.0f}) retrying...")
        time.sleep(0.2)
    print("[warn] odometry reset did not converge — proceeding anyway")


def show_idle_leds(robot: Robot) -> None:
    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 200)


def show_running_leds(robot: Robot) -> None:
    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 200)


def cancel_motion(robot: Robot, handle) -> None:
    if handle is not None:
        handle.cancel()
        handle.wait(timeout=1.0)
    robot.stop()


def print_status(robot: Robot) -> None:
    x, y, theta = robot.get_odometry_pose()
    vt = robot.get_virtual_target()
    confirmed   = robot.get_obstacle_tracks(include_unconfirmed=False)
    unconfirmed = robot.get_obstacle_tracks(include_unconfirmed=True)
    raw_pts     = robot.get_obstacles()
    vt_str = f"vt=({vt[0]:.0f}, {vt[1]:.0f})" if vt is not None else "vt=none"
    remaining = ((GOAL_X_MM - x) ** 2 + (GOAL_Y_MM - y) ** 2) ** 0.5
    print(
        f"  odom=({x:.0f}, {y:.0f}) mm  θ={theta:.1f}°  remaining={remaining:.0f} mm  "
        f"{vt_str}  raw_pts={len(raw_pts)}  unconfirmed={len(unconfirmed)}  confirmed={len(confirmed)}"
    )


def start_run(robot: Robot):
    return robot.lapf_to_goal(
        GOAL_X_MM,
        GOAL_Y_MM,
        velocity=VELOCITY_MM_S,
        tolerance=TOLERANCE_MM,
        leash_length_mm=LEASH_LENGTH_MM,
        repulsion_range_mm=REPULSION_RANGE_MM,
        target_speed_mm_s=TARGET_SPEED_MM_S,
        max_angular_rad_s=MAX_ANGULAR_RAD_S,
        repulsion_gain=REPULSION_GAIN,
        attraction_gain=ATTRACTION_GAIN,
        force_ema_alpha=FORCE_EMA_ALPHA,
        inflation_margin_mm=INFLATION_MARGIN_MM,
        leash_half_angle_deg=LEASH_HALF_ANGLE_DEG,
        blocking=False,
    )


# ---------------------------------------------------------------------------
# FSM entry point
# ---------------------------------------------------------------------------

def run(robot: Robot) -> None:
    configure_robot(robot)
    start_robot(robot)
    reset_pose(robot)

    print("=" * 56)
    print("CONE AVOIDANCE TEST — 2 m forward")
    print(f"  Goal: ({GOAL_X_MM:.0f}, {GOAL_Y_MM:.0f}) mm")
    print(f"  Speed: {VELOCITY_MM_S:.0f} mm/s  tolerance: {TOLERANCE_MM:.0f} mm")
    print(f"  Repulsion range: {REPULSION_RANGE_MM:.0f} mm  inflation: {INFLATION_MARGIN_MM:.0f} mm")
    print("=" * 56)

    show_running_leds(robot)
    motion_handle = start_run(robot)
    last_print_at = time.monotonic()

    period = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick = time.monotonic()

    while True:
        now = time.monotonic()

        if now - last_print_at >= STATUS_PRINT_INTERVAL_S:
            print_status(robot)
            last_print_at = now

        if motion_handle is not None and motion_handle.is_finished():
            robot.stop()
            show_idle_leds(robot)
            print("[DONE] goal reached")
            print_status(robot)
            return

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
