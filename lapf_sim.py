#!/usr/bin/env python3
"""
LAPF Seg2 Corridor Simulation
==============================
Simulates the robot navigating from (1525, 350) to (1525, 3350) with the
actual map cone layout.  Uses the real LeashedAPFPlanner from path_planner.py
so the numbers match the actual robot.

Usage:
    python3 lapf_sim.py [options]
    python3 lapf_sim.py --repulsion-gain 300 --attraction-gain 5
    python3 lapf_sim.py --no-plot          # text-only output

Options shown with defaults:
    --repulsion-gain  200    (LAPF_REPULSION_GAIN)
    --attraction-gain   3    (LAPF_ATTRACTION_GAIN)
    --half-angle       85    degrees (LAPF_HALF_ANGLE_DEG)
    --leash-mm        250    (LAPF_LEASH_MM)
    --repulsion-mm    430    boundary distance to start repulsion (LAPF_REPULSION_MM)
    --inflation-mm    215    robot half-width + safety margin (LAPF_INFLATION_MM)
    --velocity        120    mm/s max forward speed
    --no-plot                skip matplotlib output
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np

# ── Locate the robot package ──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOT_PKG  = os.path.join(_SCRIPT_DIR, "ros2_ws/src/robot/robot")
if _ROBOT_PKG not in sys.path:
    sys.path.insert(0, _ROBOT_PKG)

from path_planner import LeashedAPFPlanner   # the real planner

# ── Default parameters (mirror main.py) ──────────────────────────────────────

START  = (1525.0, 350.0)
GOAL   = (1525.0, 3350.0)
INIT_THETA_DEG = 90.0

DEFAULTS = dict(
    repulsion_gain  = 200.0,
    attraction_gain = 3.0,
    half_angle_deg  = 85.0,
    leash_mm        = 250.0,
    repulsion_mm    = 430.0,   # boundary-to-boundary distance at which repulsion starts
    inflation_mm    = 215.0,   # robot half-width (165) + 50 mm safety margin
    velocity        = 120.0,   # mm/s max linear speed
    max_angular     = 1.2,     # rad/s max turn rate
    target_spd      = 200.0,   # virtual-target movement speed mm/s
    ema_alpha       = 0.35,
    tolerance       = 100.0,   # goal tolerance mm
    cone_radius     = 75.0,    # tracker max-radius given to planner
    dt              = 0.04,    # 25 Hz
    max_steps       = 8000,
)

# ── Actual map cone layout ────────────────────────────────────────────────────
# Zig-zag cones (force weave through centre):
#   y=1000  x=1200  ●           (left)
#   y=1850           ●  x=1900  (right)
#   y=2800  x=1200  ●           (left)
# Side wall cones (bound the corridor on both sides):
#   left wall:   (1000,500), (1000,1500), (1000,2000)
#   right wall:  (2200,500), (2200,1000), (2200,2500)
CONES = [
    # zig-zag obstacles
    (1300, 1000),
    (1800, 1850),
    (1200, 2500),
    # left wall
    (1000, 1500),
    (1000, 2000),
    # right wall

    (2200, 1000),
    (2200, 2500),
]

# ─────────────────────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _make_planner(p: dict) -> LeashedAPFPlanner:
    tracker_lookahead = max(100.0, min(p["leash_mm"], 250.0))
    return LeashedAPFPlanner(
        max_linear        = p["velocity"],
        max_angular       = p["max_angular"],
        target_speed      = p["target_spd"],
        repulsion_gain    = p["repulsion_gain"],
        repulsion_range   = p["repulsion_mm"],
        goal_tolerance    = p["tolerance"],
        attraction_gain   = p["attraction_gain"],
        force_ema_alpha   = p["ema_alpha"],
        leash_length_mm   = p["leash_mm"],
        leash_half_angle_deg = p["half_angle_deg"],
        inflation_margin_mm  = p["inflation_mm"],
        tracker_lookahead_mm = tracker_lookahead,
    )


def _obstacle_disks(cones: list, radius: float) -> np.ndarray:
    return np.array([[cx, cy, radius] for cx, cy in cones], dtype=float)


def _debug_forces(
    vt: tuple[float, float],
    goal: tuple[float, float],
    obs: np.ndarray,
    rep_gain: float,
    attr_gain: float,
    rep_range: float,
    inflation: float,
) -> dict:
    """Reproduce the force calculation for printing."""
    vx, vy   = vt
    gx, gy   = goal
    goal_vec = np.array([gx - vx, gy - vy], dtype=float)
    gdist    = float(np.linalg.norm(goal_vec))
    attr_f   = attr_gain * (goal_vec / gdist) if gdist > 1e-6 else np.zeros(2)

    rep_f        = np.zeros(2, dtype=float)
    nearest_bnd  = float("inf")
    contributions = []
    for row in obs:
        ox, oy, rad = float(row[0]), float(row[1]), float(row[2])
        eff_r     = rad + inflation
        away      = np.array([vx - ox, vy - oy])
        cd        = float(np.linalg.norm(away))
        boundary  = cd - eff_r
        nearest_bnd = min(nearest_bnd, boundary)
        if boundary >= rep_range or cd < 1e-6:
            contributions.append((ox, oy, boundary, 0.0))
            continue
        direction = away / cd
        clearance = max(boundary, 1.0)
        mag = rep_gain * max(0.0, 1.0 / clearance - 1.0 / rep_range)
        rep_f += direction * mag
        contributions.append((ox, oy, boundary, float(mag)))

    return dict(
        attr_mag  = float(np.linalg.norm(attr_f)),
        attr_dir  = math.degrees(math.atan2(float(attr_f[1]), float(attr_f[0]))),
        rep_mag   = float(np.linalg.norm(rep_f)),
        rep_dir   = math.degrees(math.atan2(float(rep_f[1]), float(rep_f[0]))) if np.linalg.norm(rep_f) > 1e-6 else 0.0,
        net_dir   = math.degrees(math.atan2(float((attr_f + rep_f)[1]), float((attr_f + rep_f)[0]))),
        nearest   = nearest_bnd,
        per_cone  = contributions,
    )


def simulate(p: dict, cones: list, verbose: bool = True) -> tuple[list, list, list]:
    planner  = _make_planner(p)
    obs      = _obstacle_disks(cones, p["cone_radius"])
    goal     = GOAL
    gx, gy   = goal
    dt       = p["dt"]

    x, y     = START
    theta    = math.radians(INIT_THETA_DEG)

    trajectory      : list[tuple[float, float, float]] = [(x, y, theta)]
    vt_history      : list[tuple[float, float]]        = []
    force_history   : list[dict]                       = []

    # Header
    if verbose:
        print(f"\n{'Step':>5}  {'x':>7}  {'y':>7}  {'θ°':>7}  "
              f"{'vt_x':>7}  {'vt_y':>7}  {'rem_mm':>8}  "
              f"{'lin':>6}  {'ω_rad':>7}")
        print("─" * 80)

    prev_cone_prox = None   # track closest-cone transitions for narrative

    for step in range(p["max_steps"]):
        pose      = (x, y, theta)
        remaining = math.hypot(gx - x, gy - y)

        if remaining <= p["tolerance"]:
            if verbose:
                print(f"\n{'─'*80}")
                print(f"[GOAL REACHED] step={step}  t={step*dt:.1f}s  "
                      f"pos=({x:.0f}, {y:.0f})  θ={math.degrees(theta):.1f}°")
            break

        # ── Planner step ───────────────────────────────────────────────────
        linear_mm, angular_rad = planner.navigate_to_goal(pose, goal, obs, dt)
        vt = planner.get_virtual_target()
        forces = _debug_forces(
            vt, goal, obs,
            p["repulsion_gain"], p["attraction_gain"],
            p["repulsion_mm"], p["inflation_mm"],
        )

        vt_history.append(vt)
        force_history.append(forces)

        # ── Integrate unicycle kinematics ──────────────────────────────────
        x     += linear_mm * math.cos(theta) * dt
        y     += linear_mm * math.sin(theta) * dt
        theta  = _wrap(theta + angular_rad * dt)
        trajectory.append((x, y, theta))

        # ── Print every 25 steps (1 second of sim time) ───────────────────
        if verbose and step % 25 == 0:
            vx, vy = vt
            net_dir = forces["net_dir"]
            leash_angle = _leash_local_angle(pose, vt)
            print(
                f"{step:5d}  {x:7.0f}  {y:7.0f}  {math.degrees(theta):7.1f}  "
                f"{vx:7.0f}  {vy:7.0f}  {remaining:8.0f}  "
                f"{linear_mm:6.1f}  {angular_rad:7.3f}"
            )
            # Thinking narrative
            print(
                f"       ↳ attr={forces['attr_mag']:.2f}@{forces['attr_dir']:.0f}°  "
                f"rep={forces['rep_mag']:.2f}@{forces['rep_dir']:.0f}°  "
                f"net→{net_dir:.0f}°  "
                f"leash_angle={math.degrees(leash_angle):+.0f}°  "
                f"nearest_boundary={forces['nearest']:.0f}mm"
            )
            # Per-cone contribution if any are active
            active = [(ox, oy, bnd, mag) for ox, oy, bnd, mag in forces["per_cone"] if mag > 0.01]
            if active:
                for ox, oy, bnd, mag in active:
                    print(f"         cone({ox:.0f},{oy:.0f}) boundary={bnd:.0f}mm → rep_force={mag:.2f}")

    else:
        if verbose:
            print(f"\n[TIMEOUT] {p['max_steps']} steps ({p['max_steps']*dt:.0f}s) without reaching goal")

    return trajectory, vt_history, force_history


def _leash_local_angle(pose: tuple, vt: tuple) -> float:
    """Return the local-frame angle of the virtual target relative to robot heading."""
    px, py, theta = pose
    tx, ty = vt
    dx, dy = tx - px, ty - py
    local_x = math.cos(theta) * dx + math.sin(theta) * dy
    local_y = -math.sin(theta) * dx + math.cos(theta) * dy
    return math.atan2(local_y, local_x)


def print_summary(trajectory: list, vt_history: list, force_history: list, p: dict, cones: list = None) -> None:
    if cones is None:
        cones = CONES
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_dist = sum(
        math.hypot(trajectory[i][0] - trajectory[i-1][0],
                   trajectory[i][1] - trajectory[i-1][1])
        for i in range(1, len(trajectory))
    )
    duration_s = len(trajectory) * p["dt"]
    final = trajectory[-1]
    remaining = math.hypot(GOAL[0] - final[0], GOAL[1] - final[1])
    print(f"  Steps:          {len(trajectory)}")
    print(f"  Sim time:       {duration_s:.1f} s")
    print(f"  Distance:       {total_dist:.0f} mm  (straight-line: {math.hypot(GOAL[0]-START[0], GOAL[1]-START[1]):.0f} mm)")
    print(f"  Final pos:      ({final[0]:.0f}, {final[1]:.0f})  θ={math.degrees(final[2]):.1f}°")
    print(f"  Remaining:      {remaining:.0f} mm")

    # Nearest-cone approach distances
    print(f"\n  Cone closest approach:")
    for i, (cx, cy) in enumerate(cones):
        min_d = min(math.hypot(pt[0]-cx, pt[1]-cy) for pt in trajectory)
        side = "L" if cx < START[0] else "R"
        print(f"    C{i+1} ({cx:.0f},{cy:.0f}) [{side}]: {min_d:.0f} mm")


def plot(trajectory: list, vt_history: list, cones: list, p: dict) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("[matplotlib not available — skipping plot]")
        return

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 9))

    for ax in (ax_full, ax_zoom):
        # ── Cones ──────────────────────────────────────────────────────────
        for i, (cx, cy) in enumerate(cones):
            ax.add_patch(patches.Circle(
                (cx, cy), p["inflation_mm"] + p["cone_radius"],
                color="orange", alpha=0.15, zorder=1))
            ax.add_patch(patches.Circle(
                (cx, cy), p["cone_radius"],
                color="darkorange", alpha=0.9, zorder=2))
            ax.annotate(f"C{i+1}", (cx, cy), fontsize=8, ha="center", va="center",
                        color="white", fontweight="bold", zorder=3)

        # ── Virtual target path ────────────────────────────────────────────
        if vt_history:
            vxs = [v[0] for v in vt_history]
            vys = [v[1] for v in vt_history]
            ax.plot(vxs, vys, "c-", lw=0.8, alpha=0.5, label="Virtual target", zorder=3)

        # ── Robot path (colour = speed via index density) ──────────────────
        xs = [pt[0] for pt in trajectory]
        ys = [pt[1] for pt in trajectory]
        ax.plot(xs, ys, "b-", lw=1.8, label="Robot path", zorder=4)

        # Heading arrows every ~2 s
        step_skip = max(1, len(trajectory) // 30)
        for i in range(0, len(trajectory), step_skip):
            px, py, pth = trajectory[i]
            ax.annotate(
                "", xy=(px + 80*math.cos(pth), py + 80*math.sin(pth)),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
                zorder=5,
            )

        ax.plot(*START, "go", ms=10, label="Start", zorder=6)
        ax.plot(*GOAL,  "r*", ms=14, label="Goal",  zorder=6)

        # Corridor centre line
        ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
                "k--", lw=0.6, alpha=0.4, label="Corridor centre")

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal")

    ax_full.set_title("Full corridor")
    ax_full.set_xlim(600, 2400)
    ax_full.set_ylim(0, 3700)

    ax_zoom.set_title("Start zone — first 1800 mm")
    ax_zoom.set_xlim(600, 2400)
    ax_zoom.set_ylim(0, 2000)

    title = (
        f"LAPF  rep_gain={p['repulsion_gain']}  attr_gain={p['attraction_gain']}  "
        f"half_angle={p['half_angle_deg']}°  leash={p['leash_mm']} mm"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    out = os.path.join(_SCRIPT_DIR, "lapf_sim.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n[Plot saved → {out}]")
    try:
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LAPF Seg2 simulator — uses the real LeashedAPFPlanner"
    )
    parser.add_argument("--repulsion-gain",  type=float, default=DEFAULTS["repulsion_gain"])
    parser.add_argument("--attraction-gain", type=float, default=DEFAULTS["attraction_gain"])
    parser.add_argument("--half-angle",      type=float, default=DEFAULTS["half_angle_deg"])
    parser.add_argument("--leash-mm",        type=float, default=DEFAULTS["leash_mm"])
    parser.add_argument("--repulsion-mm",    type=float, default=DEFAULTS["repulsion_mm"])
    parser.add_argument("--inflation-mm",    type=float, default=DEFAULTS["inflation_mm"])
    parser.add_argument("--velocity",        type=float, default=DEFAULTS["velocity"])
    parser.add_argument("--no-plot",         action="store_true")
    args = parser.parse_args()

    p = {**DEFAULTS}
    p["repulsion_gain"]  = args.repulsion_gain
    p["attraction_gain"] = args.attraction_gain
    p["half_angle_deg"]  = args.half_angle
    p["leash_mm"]        = args.leash_mm
    p["repulsion_mm"]    = args.repulsion_mm
    p["inflation_mm"]    = args.inflation_mm
    p["velocity"]        = args.velocity

    print("=" * 80)
    print("LAPF Seg2 Corridor Simulation")
    print("=" * 80)
    print(f"  Start:           ({START[0]:.0f}, {START[1]:.0f})")
    print(f"  Goal:            ({GOAL[0]:.0f}, {GOAL[1]:.0f})")
    print(f"  Straight-line:   {math.hypot(GOAL[0]-START[0], GOAL[1]-START[1]):.0f} mm")
    print()
    print(f"  Cone layout ({len(CONES)} cones):")
    for i, (cx, cy) in enumerate(CONES):
        side = "LEFT" if cx < START[0] else "RIGHT"
        print(f"    C{i+1}  ({cx}, {cy})  {side}")
    print()
    print(f"  repulsion_gain:  {p['repulsion_gain']}")
    print(f"  attraction_gain: {p['attraction_gain']}")
    print(f"  half_angle_deg:  {p['half_angle_deg']}")
    print(f"  leash_mm:        {p['leash_mm']}")
    print(f"  repulsion_mm:    {p['repulsion_mm']}")
    print(f"  inflation_mm:    {p['inflation_mm']}")
    print(f"  velocity:        {p['velocity']} mm/s")
    print("=" * 80)

    trajectory, vt_history, force_history = simulate(p, CONES)
    print_summary(trajectory, vt_history, force_history, p)

    if not args.no_plot:
        plot(trajectory, vt_history, CONES, p)


if __name__ == "__main__":
    main()
