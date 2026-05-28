"""
lidar_viz.py — background lidar + pose visualiser
===================================================
Spawns a thread that periodically saves a matplotlib figure to
/runtime_output/lidar_viz.png (visible on the host at
ros2_ws/runtime_output/lidar_viz.png).

Open the file in any image viewer that auto-refreshes (e.g. eog, feh --auto-zoom).

Usage:
    from robot.examples.lidar_viz import LidarViz

    viz = LidarViz(robot, goal=(0.0, 3000.0))
    viz.start()
    # ... run your FSM loop ...
    viz.stop()
"""

from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot.robot import Robot

OUTPUT_PATH  = "/runtime_output/lidar_viz.png"
UPDATE_HZ    = 4.0   # frames per second saved
TRAIL_POINTS = 2000  # max robot pose history to keep

# World-frame window (mm). Adjust if your corridor is different.
VIEW_X = (-800, 800)
VIEW_Y = (-800, 3200)


class LidarViz:
    def __init__(self, robot: "Robot", goal: tuple[float, float] = (0.0, 3000.0)):
        self._robot  = robot
        self._goal   = goal
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stop   = threading.Event()
        self._trail: list[tuple[float, float, float]] = []  # (x, y, theta_deg)

    def start(self) -> None:
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        interval = 1.0 / UPDATE_HZ
        t0       = time.monotonic()
        frame    = 0

        while not self._stop.is_set():
            tick_start = time.monotonic()
            try:
                self._save_frame(plt, patches, time.monotonic() - t0, frame)
                frame += 1
            except Exception as exc:
                print(f"[lidar_viz] frame error: {exc}")
            elapsed = time.monotonic() - tick_start
            self._stop.wait(max(0.0, interval - elapsed))

    def _save_frame(self, plt, patches, elapsed_s: float, frame: int) -> None:
        robot = self._robot

        # ── Sample robot state ────────────────────────────────────────────
        x, y, theta_deg = robot.get_pose()
        theta_rad = math.radians(theta_deg)
        vt        = robot.get_virtual_target()       # world frame (mm) or None
        raw_robot = robot.get_obstacles()            # robot frame [(x,y), ...]
        tracks    = robot.get_obstacle_tracks(include_unconfirmed=False)

        # Accumulate trail
        self._trail.append((x, y, theta_deg))
        if len(self._trail) > TRAIL_POINTS:
            self._trail = self._trail[-TRAIL_POINTS:]

        # Transform raw lidar points → world frame
        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        raw_world = [
            (x + px * ct - py * st, y + px * st + py * ct)
            for px, py in raw_robot
        ]

        # ── Build figure ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 12))
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(*VIEW_Y)
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.set_xlabel("X (mm)", color="white")
        ax.set_ylabel("Y (mm)", color="white")
        ax.grid(True, color="#333333", linewidth=0.5)

        # Raw lidar points
        if raw_world:
            wxs, wys = zip(*raw_world)
            ax.scatter(wxs, wys, s=3, c="#ff6600", alpha=0.5, zorder=2, label="lidar pts")

        # Confirmed obstacle tracks
        for t in tracks:
            cx, cy, r = float(t["x"]), float(t["y"]), float(t["radius"])
            ax.add_patch(patches.Circle(
                (cx, cy), r, color="#ff4444", alpha=0.25, zorder=3))
            ax.add_patch(patches.Circle(
                (cx, cy), r, fill=False, edgecolor="#ff4444", linewidth=1.0, zorder=3))

        # Robot trail
        if len(self._trail) > 1:
            txs = [p[0] for p in self._trail]
            tys = [p[1] for p in self._trail]
            ax.plot(txs, tys, color="#4488ff", linewidth=1.2,
                    alpha=0.6, zorder=4, label="trajectory")

        # Virtual target
        if vt is not None:
            ax.plot(vt[0], vt[1], "c+", ms=14, mew=2.5, zorder=6, label="virtual target")
            ax.plot([x, vt[0]], [y, vt[1]], color="cyan", linewidth=0.8,
                    alpha=0.4, zorder=5, linestyle="--")

        # Goal
        ax.plot(*self._goal, "r*", ms=16, zorder=7, label="goal")

        # Robot body + heading arrow
        body_r = 165
        ax.add_patch(patches.Circle(
            (x, y), body_r, color="#22cc44", alpha=0.35, zorder=8))
        ax.add_patch(patches.Circle(
            (x, y), body_r, fill=False, edgecolor="#22cc44", linewidth=1.5, zorder=8))
        arrow_len = 220
        ax.annotate(
            "", xy=(x + arrow_len * ct, y + arrow_len * st), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="#22ff55", lw=2.5),
            zorder=9,
        )

        # Title
        rem = math.hypot(self._goal[0] - x, self._goal[1] - y)
        ax.set_title(
            f"t={elapsed_s:.1f}s   pose=({x:.0f}, {y:.0f})   θ={theta_deg:.1f}°   rem={rem:.0f}mm"
            f"\nconf={len(tracks)}  raw={len(raw_robot)}",
            color="white", fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7, facecolor="#222222",
                  labelcolor="white", framealpha=0.7)

        fig.tight_layout()
        fig.savefig(OUTPUT_PATH, dpi=90, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
