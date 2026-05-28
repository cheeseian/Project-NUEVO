"""
lidar_gap_steer.py — direct lidar gap-finding obstacle avoidance
================================================================
No obstacle tracker, no confirmation delay, no force accumulation.

Each tick:
  1. Grab raw lidar points (robot body frame, +x=forward, +y=left).
  2. Filter to forward hemisphere (±FOV_HALF_DEG) within REACT_RANGE_MM.
  3. Bin into N_SECTORS angular buckets, compute weighted density per bucket.
  4. Hard-block any sector whose min lidar range < BLOCK_DIST_MM.
  5. Among unblocked sectors score = open_score * goal_score
       open_score = 1 / (density + 1)
       goal_score = Gaussian centred on goal bearing
  6. EMA-smooth the chosen steer angle (kills jitter).
  7. Speed scales with forward clearance and steer magnitude.

HOW TO RUN
----------
    ROBOT_FSM_MODULE=robot.examples.lidar_gap_steer ./run_robot.sh
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from robot.robot import FirmwareState, Robot, Unit
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
# Hardware — must match main.py
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
# Mission
# ---------------------------------------------------------------------------

GOAL_X_MM    = 0.0
GOAL_Y_MM    = 3000.0
TOLERANCE_MM = 120.0

# ---------------------------------------------------------------------------
# Gap-steering parameters
# ---------------------------------------------------------------------------

N_SECTORS      = 24       # angular bins covering ±FOV_HALF_DEG
FOV_HALF_DEG   = 90.0     # half-width of scanned forward arc

REACT_RANGE_MM   = 800.0  # ignore lidar points beyond this
MIN_RANGE_MM     = 80.0   # ignore lidar points closer than this (own chassis)

# Treat each lidar point as a disk of this radius before sector assignment.
# Expands each point's angular footprint and reduces its effective range,
# so cones near sector boundaries still block the adjacent sector and
# the robot keeps more clearance without changing BLOCK_DIST_MM.
POINT_INFLATE_MM = 120.0

# Hard-block: sectors where a lidar point is within this range are completely
# excluded from consideration, regardless of goal direction.
# = robot half-width (165) + cone surface margin + reaction buffer
BLOCK_DIST_MM  = 350.0

# When a sector is blocked, also penalise its immediate neighbours so the
# robot doesn't clip a cone that straddles a sector boundary.
BLOCK_SPREAD   = 1         # number of adjacent sectors to soft-penalise each side

# Speed control — only look directly ahead so lateral cones don't kill forward speed
SPEED_CONE_DEG = 20.0     # half-angle of forward cone for min-range check
STOP_DIST_MM   = 180.0    # full stop only if something is this close dead ahead
CRUISE_DIST_MM = 450.0    # full speed when forward cone is clear beyond this

MAX_SPEED_MM_S = 130.0
MAX_ANGULAR    = 1.5      # rad/s cap
K_ANGULAR      = 1.8      # P-gain: rad/s per radian of steer angle

# Steering smoothing — kills scan-to-scan jitter (same role as force_ema_alpha)
EMA_ALPHA      = 0.25     # weight on new measurement; lower = smoother/less jitter

# Goal preference
GOAL_SIGMA_DEG = 40.0     # Gaussian width around goal bearing
GOAL_WEIGHT    = 1.2      # exponent on goal score (lower = less likely to tunnel)

# Scan mode — engaged when speed drops near zero (all forward sectors blocked)
# Robot creeps forward slowly while rotating to find the opening.
SCAN_THRESHOLD_MM_S = 20.0   # enter scan when linear drops below this
SCAN_CREEP_MM_S     = 45.0   # slow forward speed while scanning — keeps momentum
SCAN_RATE_DEG_S     = 28.0   # rotation speed while scanning
SCAN_HOLD_TICKS     = 100    # ticks (~2 s at 50 Hz) before scan direction may flip

STATUS_INTERVAL_S = 0.5

# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _gap_steer(
    robot_pts: list[tuple[float, float]],
    goal_angle_local: float,
    prev_steer: float,
) -> tuple[float, float, float, list]:
    """
    Returns (linear_mm_s, angular_rad_s, smoothed_steer_rad, sector_debug).
    sector_debug: list of dicts, one per sector, for printing.
    prev_steer: EMA state carried between ticks.
    """
    fov_half  = math.radians(FOV_HALF_DEG)
    sector_w  = 2.0 * fov_half / N_SECTORS
    centers   = np.linspace(
        -fov_half + sector_w / 2.0,
         fov_half - sector_w / 2.0,
        N_SECTORS,
    )

    density   = np.zeros(N_SECTORS, dtype=float)
    min_range = np.full(N_SECTORS, REACT_RANGE_MM, dtype=float)

    speed_cone = math.radians(SPEED_CONE_DEG)
    fwd_min    = REACT_RANGE_MM

    for x_mm, y_mm in robot_pts:
        r = math.hypot(x_mm, y_mm)
        if r < MIN_RANGE_MM or r > REACT_RANGE_MM:
            continue
        a = math.atan2(y_mm, x_mm)   # robot frame: +y=left → +angle=left
        if abs(a) > fov_half:
            continue

        # Inflate the point: treat it as a disk of POINT_INFLATE_MM radius.
        # effective_r is the range to the near edge of that disk (what the
        # robot's body would actually reach first).
        effective_r = max(r - POINT_INFLATE_MM, 1.0)

        # Angular half-span the inflated disk subtends from the robot
        half_span = math.atan2(POINT_INFLATE_MM, max(effective_r, 1.0))

        # Assign contribution to every sector overlapping the inflated arc
        i_lo = max(0,           int((a - half_span + fov_half) / sector_w))
        i_hi = min(N_SECTORS-1, int((a + half_span + fov_half) / sector_w))
        weight = REACT_RANGE_MM / (r + 1.0)
        for idx in range(i_lo, i_hi + 1):
            density[idx] += weight
            if effective_r < min_range[idx]:
                min_range[idx] = effective_r

        if abs(a) < speed_cone and effective_r < fwd_min:
            fwd_min = effective_r

    # Score sectors
    goal_sigma = math.radians(GOAL_SIGMA_DEG)
    scores = np.zeros(N_SECTORS, dtype=float)
    open_scores = np.zeros(N_SECTORS, dtype=float)
    goal_scores = np.zeros(N_SECTORS, dtype=float)
    blocked = np.zeros(N_SECTORS, dtype=bool)

    # Primary block pass
    for i in range(N_SECTORS):
        if min_range[i] < BLOCK_DIST_MM:
            blocked[i] = True

    # Spread block to adjacent sectors so cone edges don't slip through
    spread_blocked = blocked.copy()
    for i in range(N_SECTORS):
        if blocked[i]:
            for j in range(max(0, i - BLOCK_SPREAD), min(N_SECTORS, i + BLOCK_SPREAD + 1)):
                spread_blocked[j] = True
    blocked = spread_blocked

    any_open = False
    for i, c in enumerate(centers):
        if blocked[i]:
            continue
        open_scores[i] = 1.0 / (density[i] + 1.0)
        angle_diff = _wrap(c - goal_angle_local)
        goal_scores[i] = math.exp(-0.5 * (angle_diff / goal_sigma) ** 2)
        scores[i] = open_scores[i] * (goal_scores[i] ** GOAL_WEIGHT)
        any_open = True

    fallback = not any_open
    if fallback:
        scores = 1.0 / (density + 1.0)

    best_idx    = int(np.argmax(scores))
    raw_steer   = float(centers[best_idx])

    # EMA smooth to kill scan-to-scan jitter
    smoothed    = EMA_ALPHA * raw_steer + (1.0 - EMA_ALPHA) * prev_steer

    # Angular command
    angular = float(np.clip(K_ANGULAR * smoothed, -MAX_ANGULAR, MAX_ANGULAR))

    # Speed: forward clearance factor × turn magnitude factor
    t = (fwd_min - STOP_DIST_MM) / max(1.0, CRUISE_DIST_MM - STOP_DIST_MM)
    speed_factor = max(0.0, min(1.0, t))
    turn_factor  = max(0.25, 1.0 - abs(smoothed) / math.pi)
    linear = MAX_SPEED_MM_S * speed_factor * turn_factor

    # Build sector debug list
    sector_debug = []
    for i, c in enumerate(centers):
        sector_debug.append({
            "angle_deg": math.degrees(c),
            "density":   round(density[i], 1),
            "min_range": round(min_range[i]),
            "blocked":   bool(blocked[i]),
            "score":     round(scores[i], 3),
            "best":      i == best_idx,
        })

    return linear, angular, smoothed, sector_debug


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _goal_bearing_local(pose_rad: tuple, goal: tuple) -> float:
    x, y, theta = pose_rad
    dx, dy = goal[0] - x, goal[1] - y
    return _wrap(math.atan2(dy, dx) - theta)

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
            p = robot.get_odometry_parameters()
            print(f"[gap_steer] odom confirmed (attempt {attempt+1}): "
                  f"L=M{p['left_motor_number']} R=M{p['right_motor_number']}")
            return True
        print(f"[gap_steer] odom not confirmed (attempt {attempt+1}), retrying…")
        time.sleep(0.2)
    return False


def _go_running(robot: Robot) -> None:
    for attempt in range(5):
        if robot.set_state(FirmwareState.RUNNING, timeout=10.0):
            print(f"[gap_steer] firmware → RUNNING (attempt {attempt+1})")
            return
        print(f"[gap_steer] set_state RUNNING failed (attempt {attempt+1}), retrying…")
        time.sleep(1.0)
    print("[gap_steer] WARNING: could not confirm RUNNING — continuing anyway")


def _reset_odom(robot: Robot) -> None:
    time.sleep(0.5)
    robot.reset_odometry()
    robot.wait_for_odometry_reset(timeout=3.0)
    x, y, theta = robot.get_pose()
    print(f"[gap_steer] odom reset  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")
    if abs(theta - INITIAL_THETA_DEG) > 5.0:
        robot.reset_odometry()
        robot.wait_for_odometry_reset(timeout=3.0)
        x, y, theta = robot.get_pose()
        print(f"[gap_steer] re-reset  pose=({x:.0f},{y:.0f}) θ={theta:.1f}°")

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
        print("[gap_steer] stopped")


def _run(robot: Robot) -> None:
    _configure(robot)

    if not _set_odom_params(robot):
        print("[gap_steer] FATAL: odom params never confirmed — aborting")
        return

    _go_running(robot)

    robot.set_led(LED.GREEN, 0)
    robot.set_led(LED.ORANGE, 255)

    print()
    print("=" * 60)
    print("LIDAR GAP STEERING — corridor entrance → 3000 mm ahead")
    print(f"  Goal: ({GOAL_X_MM:.0f}, {GOAL_Y_MM:.0f}) mm   speed: {MAX_SPEED_MM_S:.0f} mm/s")
    print(f"  Sectors: {N_SECTORS}  FOV: ±{FOV_HALF_DEG:.0f}°  react: {REACT_RANGE_MM:.0f}mm")
    print(f"  block_dist={BLOCK_DIST_MM:.0f}mm  ema={EMA_ALPHA}  goal_sigma={GOAL_SIGMA_DEG}°  goal_weight={GOAL_WEIGHT}")
    print()
    print("  Place robot at corridor entrance, then press Enter to start.")
    print("  Ctrl-C to abort.")
    print("=" * 60)

    _enter = threading.Event()
    def _wait():
        try:
            input()
        except EOFError:
            pass
        _enter.set()

    threading.Thread(target=_wait, daemon=True).start()
    _enter.wait()

    _reset_odom(robot)
    robot.set_led(LED.ORANGE, 0)
    robot.set_led(LED.GREEN, 255)
    print("[gap_steer] starting…")

    goal           = (GOAL_X_MM, GOAL_Y_MM)
    period         = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick      = time.monotonic()
    last_status    = time.monotonic()
    steer_state    = 0.0   # EMA carry
    scan_dir       = 0.0   # +1=rotate left, -1=rotate right, 0=not scanning
    scan_hold      = 0     # countdown: ticks remaining before scan direction may flip

    while True:
        now          = time.monotonic()
        pose         = robot.get_pose()        # (x, y, theta_deg)
        x, y, theta_deg = pose
        theta        = math.radians(theta_deg)
        pose_rad     = (x, y, theta)

        remaining = math.hypot(GOAL_X_MM - x, GOAL_Y_MM - y)
        if remaining <= TOLERANCE_MM:
            robot.stop()
            robot.set_led(LED.GREEN, 0)
            robot.set_led(LED.ORANGE, 255)
            print(f"[gap_steer] DONE  pose=({x:.0f},{y:.0f}) θ={theta_deg:.1f}°")
            return

        raw_pts      = robot.get_obstacles()   # robot-frame (x_mm, y_mm) list
        goal_bearing = _goal_bearing_local(pose_rad, goal)
        linear, angular, steer_state, sector_debug = _gap_steer(raw_pts, goal_bearing, steer_state)

        # ── Scan mode ─────────────────────────────────────────────────────────
        # When the forward path is blocked and speed collapses, stop twitching
        # and instead rotate slowly toward whichever half has the best opening.
        if linear < SCAN_THRESHOLD_MM_S:
            # Use the globally best sector to decide direction, not a left/right split.
            # This prevents the robot from getting stuck when one side is slightly
            # better but the real gap is further around on the other side.
            best_s    = max(sector_debug, key=lambda s: s["score"])
            preferred = 1.0 if best_s["angle_deg"] > 0 else -1.0

            if scan_hold <= 0 or scan_dir == 0.0:
                scan_dir  = preferred
                scan_hold = SCAN_HOLD_TICKS
            else:
                scan_hold -= 1

            linear_cmd  = SCAN_CREEP_MM_S
            angular_cmd = scan_dir * SCAN_RATE_DEG_S
            mode = f"SCAN({'L' if scan_dir > 0 else 'R'}) hold={scan_hold}"
        else:
            scan_dir  = 0.0
            scan_hold = 0
            linear_cmd  = linear
            angular_cmd = math.degrees(angular)
            mode = "NAV"

        robot.set_velocity(linear_cmd, angular_cmd)

        if now - last_status >= STATUS_INTERVAL_S:
            print(
                f"  [{mode}] pose=({x:.0f},{y:.0f}) θ={theta_deg:.1f}°  rem={remaining:.0f}mm"
                f"  steer={math.degrees(steer_state):+.1f}°  lin={linear_cmd:.0f}  ω={angular_cmd:.1f}°/s"
                f"  goal_bear={math.degrees(goal_bearing):+.1f}°  pts={len(raw_pts)}"
            )
            header = f"  {'ang':>6} {'minR':>5} {'dens':>6} {'score':>6}"
            print(header)
            for s in sector_debug:
                flag = "◀BEST" if s["best"] else ("BLOCK" if s["blocked"] else "")
                print(
                    f"  {s['angle_deg']:>+6.1f}°"
                    f" {s['min_range']:>5}mm"
                    f" {s['density']:>6.1f}"
                    f" {s['score']:>6.3f}"
                    f"  {flag}"
                )
            last_status = now

        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
