"""
teleop.py — arrow-key teleoperation
=====================================
Drive the robot manually from your terminal.  Use run_teleop.sh to launch.

CONTROLS
--------
  ↑  Forward       ↓  Backward
  ←  Turn left     →  Turn right
  q / Ctrl+C        Quit
  (no key held)     Auto-stop after 150 ms
"""

from __future__ import annotations

import sys
import tty
import termios
import threading
import time

from robot.robot import FirmwareState, Robot, Unit
from robot.hardware_map import DEFAULT_FSM_HZ, LED, Motor

# ── Hardware config (must match main.py) ─────────────────────────────────────
POSITION_UNIT            = Unit.MM
WHEEL_DIAMETER           = 74.0
WHEEL_BASE               = 321.0
INITIAL_THETA_DEG        = 90.0
LEFT_WHEEL_MOTOR         = Motor.DC_M2
LEFT_WHEEL_DIR_INVERTED  = False
RIGHT_WHEEL_MOTOR        = Motor.DC_M1
RIGHT_WHEEL_DIR_INVERTED = True

# ── Teleop parameters ─────────────────────────────────────────────────────────
LINEAR_SPEED_MM_S   = 120.0   # mm/s forward / backward
ANGULAR_SPEED_DEG_S = 70.0    # deg/s left / right
KEY_TIMEOUT_S       = 0.15    # stop if no key received for this long

# ── Shared key state ──────────────────────────────────────────────────────────
_lock        = threading.Lock()
_state       = {"key": "", "at": 0.0}  # key: UP/DOWN/LEFT/RIGHT/QUIT/""
_stop_reader = threading.Event()


def _read_keys() -> None:
    """Background thread: decode arrow-key escape sequences from the terminal."""
    try:
        tty_file = open("/dev/tty", "rb", buffering=0)
    except OSError:
        tty_file = sys.stdin.buffer

    fd = tty_file.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except termios.error:
        print("\r\n[teleop] stdin is not a TTY — run with run_teleop.sh (docker exec -it)\r\n")
        with _lock:
            _state["key"] = "QUIT"
        return

    try:
        tty.setraw(fd)
        while not _stop_reader.is_set():
            ch = tty_file.read(1)
            if not ch:
                break
            if ch == b"\x1b":
                rest = tty_file.read(2)
                seq = ch + rest
                with _lock:
                    if seq == b"\x1b[A":
                        _state["key"] = "UP"
                    elif seq == b"\x1b[B":
                        _state["key"] = "DOWN"
                    elif seq == b"\x1b[C":
                        _state["key"] = "RIGHT"
                    elif seq == b"\x1b[D":
                        _state["key"] = "LEFT"
                    else:
                        _state["key"] = ""
                    _state["at"] = time.monotonic()
            elif ch in (b"q", b"Q", b"\x03"):
                with _lock:
                    _state["key"] = "QUIT"
                break
            else:
                with _lock:
                    _state["key"] = ""
                    _state["at"] = time.monotonic()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def run(robot: Robot) -> None:
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

    for attempt in range(5):
        if robot.set_state(FirmwareState.RUNNING, timeout=10.0):
            print(f"[teleop] firmware → RUNNING (attempt {attempt + 1})")
            break
        time.sleep(1.0)

    robot.reset_odometry()
    robot.wait_for_odometry_reset(timeout=3.0)

    print("\r\n" + "=" * 52)
    print("  TELEOP — arrow keys to drive, q to quit")
    print("  ↑ Forward  ↓ Backward  ← Turn left  → Turn right")
    print("=" * 52 + "\r\n")

    key_thread = threading.Thread(target=_read_keys, daemon=True)
    key_thread.start()

    robot.set_led(LED.GREEN, 255)
    robot.set_led(LED.ORANGE, 0)

    period       = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick    = time.monotonic()
    last_display = ""

    _LABELS = {
        "UP":    "↑ Forward   ",
        "DOWN":  "↓ Backward  ",
        "LEFT":  "← Left      ",
        "RIGHT": "→ Right     ",
        "":      "◼ Stopped   ",
    }

    try:
        while True:
            now = time.monotonic()
            with _lock:
                raw_key = _state["key"]
                key_age = now - _state["at"]

            if raw_key == "QUIT":
                break

            # Auto-stop when key has not been refreshed recently
            key = raw_key if key_age < KEY_TIMEOUT_S else ""

            label = _LABELS.get(key, "◼ Stopped   ")
            if label != last_display:
                print(f"\r{label}", end="", flush=True)
                last_display = label

            if key == "UP":
                robot.set_velocity(LINEAR_SPEED_MM_S, 0.0)
            elif key == "DOWN":
                robot.set_velocity(-LINEAR_SPEED_MM_S, 0.0)
            elif key == "LEFT":
                robot.set_velocity(0.0, ANGULAR_SPEED_DEG_S)
            elif key == "RIGHT":
                robot.set_velocity(0.0, -ANGULAR_SPEED_DEG_S)
            else:
                robot.stop()

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    finally:
        _stop_reader.set()
        robot.stop()
        robot.set_led(LED.GREEN, 0)
        robot.set_led(LED.ORANGE, 0)
        print("\r\n[teleop] motors stopped.")
