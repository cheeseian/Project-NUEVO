from __future__ import annotations
import math
import time

from robot.robot import FirmwareState, Robot, Unit
from robot.hardware_map import (
    Button, DEFAULT_FSM_HZ, LED, Motor,
    LIDAR_FOV_DEG, LIDAR_MOUNT_THETA_DEG,
    LIDAR_MOUNT_X_MM, LIDAR_MOUNT_Y_MM,
    LIDAR_RANGE_MAX_MM, LIDAR_RANGE_MIN_MM,
)
from robot.util import densify_polyline

# ---------------------------------------------------------------------------
# Robot hardware configuration
# ---------------------------------------------------------------------------

POSITION_UNIT        = Unit.MM
WHEEL_DIAMETER       = 74.0
WHEEL_BASE           = 321.0
INITIAL_THETA_DEG    = 90.0

LEFT_WHEEL_MOTOR         = Motor.DC_M2
LEFT_WHEEL_DIR_INVERTED  = False
RIGHT_WHEEL_MOTOR        = Motor.DC_M1
RIGHT_WHEEL_DIR_INVERTED = True

# ---------------------------------------------------------------------------
# Pure Pursuit parameters (segments 1 and 3 — no cones)
# ---------------------------------------------------------------------------

PP_LOOKAHEAD_MM   = 100.0
PP_MAX_LINEAR     = 140.0
PP_MAX_ANGULAR    = 1.5
PP_GOAL_TOL       = 20.0
PP_ALPHA_LD       = 0.7
PP_X_L            = 300.0
PP_OFFSET         = 270.0
PP_LANE_WIDTH     = 500.0

# ---------------------------------------------------------------------------
# LAPF parameters — cone avoidance segment (1220, 305) → (1220, 3350)
# ---------------------------------------------------------------------------

LAPF_GOAL_X_MM       = 1220.0
LAPF_GOAL_Y_MM       = 3350.0
LAPF_VELOCITY_MM_S   = 130.0
LAPF_TOLERANCE_MM    = 100.0
LAPF_MAX_ANGULAR     = 1.2
LAPF_LEASH_MM        = 220.0
LAPF_HALF_ANGLE_DEG  = 50.0
LAPF_REPULSION_MM    = 500.0
LAPF_INFLATION_MM    = 260.0
LAPF_TARGET_SPD_MM_S = 200.0
LAPF_REPULSION_GAIN  = 650.0
LAPF_ATTRACTION_GAIN = 1.0
LAPF_EMA_ALPHA       = 0.35

STATUS_INTERVAL_S    = 0.5

# ---------------------------------------------------------------------------
# Map path — split at cone corridor boundaries
# ---------------------------------------------------------------------------

# Segment 1: start → entry of cone corridor
PATH_SEG1_CTRL = [
    (   0.0,    0.0),
    (   0.0, 3350.0),
    ( 610.0, 3350.0),
    ( 610.0,  305.0),
    (1220.0,  305.0),
]

# Segment 3: exit of cone corridor → finish
PATH_SEG3_CTRL = [
    (1220.0, 3350.0),
    (2440.0, 3350.0),
    (2440.0,    0.0),
    (2745.0,    0.0),
]


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


def init_pp(robot: Robot, ctrl_points: list, spacing: float = 300.0) -> None:
    robot._nav_follow_pp_path(
        lookahead_distance=PP_LOOKAHEAD_MM,
        max_linear_speed=PP_MAX_LINEAR,
        max_angular_speed=PP_MAX_ANGULAR,
        goal_tolerance=PP_GOAL_TOL,
        obstacles_range=450.0,
        view_angle=math.radians(70.0),
        safe_dist=250.0,
        avoidance_delay=150,
        alpha_Ld=PP_ALPHA_LD,
        offset=PP_OFFSET,
        lane_width=PP_LANE_WIDTH,
        obstacle_avoidance=False,
        x_L=PP_X_L,
    )
    path = densify_polyline(ctrl_points, spacing=spacing)
    robot.planner.set_path(path)


def start_lapf(robot: Robot):
    return robot.lapf_to_goal(
        LAPF_GOAL_X_MM,
        LAPF_GOAL_Y_MM,
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


def show_idle_leds(robot: Robot) -> None:
    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 255)


def show_moving_leds(robot: Robot) -> None:
    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 255)


def print_status(robot: Robot, label: str = "") -> None:
    x, y, theta = robot.get_odometry_pose()
    confirmed = robot.get_obstacle_tracks(include_unconfirmed=False)
    raw_pts   = robot.get_obstacles()
    tag = f"[{label}]" if label else ""
    print(f"{tag}  odom=({x:.0f}, {y:.0f}) mm  θ={theta:.1f}°"
          f"  raw_pts={len(raw_pts)}  confirmed={len(confirmed)}")


# ---------------------------------------------------------------------------
# FSM entry point
# ---------------------------------------------------------------------------

def run(robot: Robot) -> None:
    configure_robot(robot)

    robot.set_state(FirmwareState.RUNNING)
    robot.reset_odometry()
    robot.wait_for_pose_update(timeout=0.5)

    period = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick = time.monotonic()
    last_status_at = time.monotonic()

    # Pre-build segment 1 path so IDLE state is ready immediately
    init_pp(robot, PATH_SEG1_CTRL)

    print("=" * 60)
    print("MAP RUN — PP × LAPF")
    print("  Seg1: (0,0) → (1220,305)   [Pure Pursuit]")
    print("  Seg2: (1220,305) → (1220,3350)  [LAPF cone avoidance]")
    print("  Seg3: (1220,3350) → (2745,0)  [Pure Pursuit]")
    print("  BTN_1 = start   BTN_2 = stop")
    print("=" * 60)

    state = "PP_SEG1"
    lapf_handle = None
    show_moving_leds(robot)

    while True:
        now = time.monotonic()

        # ------------------------------------------------------------------
        if state == "IDLE":
            show_idle_leds(robot)
            if now - last_status_at >= STATUS_INTERVAL_S:
                robot._draw_lidar_obstacles()
                last_status_at = now
            if robot.get_button(Button.BTN_1):
                print("[FSM] IDLE → PP_SEG1")
                show_moving_leds(robot)
                state = "PP_SEG1"

        # ------------------------------------------------------------------
        elif state == "PP_SEG1":
            if now - last_status_at >= STATUS_INTERVAL_S:
                print_status(robot, "PP_SEG1")
                last_status_at = now
            result = robot._nav_follow_pp_path_loop()
            if result == "IDLE":
                print("[FSM] PP_SEG1 done → starting LAPF cone segment")
                lapf_handle = start_lapf(robot)
                state = "LAPF_SEG2"

        # ------------------------------------------------------------------
        elif state == "LAPF_SEG2":
            if now - last_status_at >= STATUS_INTERVAL_S:
                x, y, theta = robot.get_odometry_pose()
                vt = robot.get_virtual_target()
                confirmed   = robot.get_obstacle_tracks(include_unconfirmed=False)
                unconfirmed = robot.get_obstacle_tracks(include_unconfirmed=True)
                raw_pts     = robot.get_obstacles()
                vt_str = f"vt=({vt[0]:.0f},{vt[1]:.0f})" if vt is not None else "vt=none"
                remaining = ((LAPF_GOAL_X_MM - x) ** 2 + (LAPF_GOAL_Y_MM - y) ** 2) ** 0.5
                print(f"[LAPF_SEG2]  odom=({x:.0f},{y:.0f}) θ={theta:.1f}°"
                      f"  rem={remaining:.0f} mm  {vt_str}"
                      f"  raw={len(raw_pts)}  unc={len(unconfirmed)}  conf={len(confirmed)}")
                last_status_at = now
            if lapf_handle is not None and lapf_handle.is_finished():
                robot.stop()
                print("[FSM] LAPF_SEG2 done → PP_SEG3")
                init_pp(robot, PATH_SEG3_CTRL)
                show_moving_leds(robot)
                state = "PP_SEG3"

        # ------------------------------------------------------------------
        elif state == "PP_SEG3":
            if now - last_status_at >= STATUS_INTERVAL_S:
                print_status(robot, "PP_SEG3")
                last_status_at = now
            result = robot._nav_follow_pp_path_loop()
            if result == "IDLE":
                robot.stop()
                show_idle_leds(robot)
                print("[FSM] PP_SEG3 done — run complete!")
                print_status(robot, "DONE")
                return

        # ------------------------------------------------------------------
        if robot.get_button(Button.BTN_2):
            robot.stop()
            show_idle_leds(robot)
            print("[FSM] BTN_2 — aborted")
            return

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
