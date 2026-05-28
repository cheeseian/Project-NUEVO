#!/usr/bin/env python3
"""LiDAR point cloud visualizer — reads piped x y data from scan_dump.py via stdin."""
import sys
import queue
import threading
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

q: queue.Queue = queue.Queue(maxsize=2)


def _reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            vals = np.fromstring(line, dtype=np.float32, sep=' ')
            if len(vals) >= 2 and len(vals) % 2 == 0:
                pts = vals.reshape(-1, 2)
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put(pts)
        except Exception:
            pass


threading.Thread(target=_reader, daemon=True).start()

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')
ax.set_xlim(-3500, 3500)
ax.set_ylim(-3500, 3500)
ax.set_aspect('equal')
ax.set_xlabel('X (mm)', color='white')
ax.set_ylabel('Y (mm)', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

for r_mm in (500, 1000, 2000, 3000):
    ax.add_patch(plt.Circle((0, 0), r_mm, color='#2a2a2a', fill=False, linewidth=0.8))
    ax.text(r_mm + 50, 50, f'{r_mm}mm', color='#555', fontsize=7)

ax.axhline(0, color='#2a2a2a', linewidth=0.6)
ax.axvline(0, color='#2a2a2a', linewidth=0.6)
ax.plot(0, 0, 'r+', markersize=18, markeredgewidth=2.5, zorder=5, label='robot')

scatter = ax.scatter([], [], s=3, c='#00ff88', alpha=0.9, linewidths=0)
title = ax.set_title('LiDAR /scan  —  waiting for data…', color='white', fontsize=11)
fig.canvas.manager.set_window_title('LiDAR Viewer')


def _update(_frame):
    try:
        pts = q.get_nowait()
        scatter.set_offsets(pts)
        title.set_text(f'LiDAR /scan  —  {len(pts)} pts')
    except queue.Empty:
        pass
    return scatter, title


ani = animation.FuncAnimation(
    fig, _update, interval=50, blit=True, cache_frame_data=False
)
plt.tight_layout()
plt.show()
