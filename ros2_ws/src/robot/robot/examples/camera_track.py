"""
camera_track.py — blue-target tracking with pan servo and pitch motor
======================================================================
Reads the camera, detects a #007FFF blue target, and drives:
  - Servo CH 16  (pan  — left/right)  via set_servo()     0–180°
  - Motor M3 PWM (pitch — up/down)    via set_motor_pwm()  ±200 (pulse)

When no target is visible the head sweeps left/right (scan mode).
Annotated live feed served at http://<pi-ip>:8082

Run inside ROS2:
    ros2 run robot camera_track

Or standalone (launches its own node):
    python3 -m robot.examples.camera_track
"""

from __future__ import annotations

import math
import signal
import socket
import subprocess
import threading
import time

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

from robot.robot import Robot
from robot.hardware_map import DEFAULT_FSM_HZ, DCMotorMode

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

_CAM_DEVICE     = "/dev/video10"
_CAM_WIDTH      = 640
_CAM_HEIGHT     = 480
_CAM_FPS        = 30
_FRAME_BYTES    = _CAM_WIDTH * _CAM_HEIGHT * 3  # bgr24

# ---------------------------------------------------------------------------
# Blue target: #007FFF = RGB(0, 127, 255) → HSV H ≈ 105 (OpenCV 0–179 scale)
# ---------------------------------------------------------------------------

_HSV_LOW  = np.array([ 95, 120,  80], dtype=np.uint8)
_HSV_HIGH = np.array([115, 255, 255], dtype=np.uint8)
_MIN_BLOB_AREA = 500  # px² — ignore noise

# ---------------------------------------------------------------------------
# Pan servo (CH 16)
# ---------------------------------------------------------------------------

PAN_CHANNEL    = 16
PAN_CENTER_DEG = 90.0
PAN_MIN_DEG    = 0.0
PAN_MAX_DEG    = 180.0
PAN_SCAN_STEP  = 0.4    # deg per tick while scanning (no target)

# ---------------------------------------------------------------------------
# Step-based tracking — one fixed step every STEP_INTERVAL_S
# ---------------------------------------------------------------------------

STEP_INTERVAL_S  = 0.2   # seconds between correction steps
COARSE_THRESH_PX = 200   # px — use coarse step when error exceeds this

PAN_STEP_COARSE  = 4.0   # deg per step when error > COARSE_THRESH_PX
PAN_STEP_FINE    = 0.5   # deg per step when error <= COARSE_THRESH_PX
CENTER_TOL_PX    = 10    # px — no step taken when within this of centre

PITCH_MOTOR = 3
PITCH_PWM   = 250   # full drive — motor won't move below this

# ---------------------------------------------------------------------------
# Image brightness/contrast boost applied to every frame
#   alpha > 1.0 = more contrast, beta > 0 = brighter (0–255 shift)
# ---------------------------------------------------------------------------

CAM_ALPHA = 1.4   # contrast multiplier
CAM_BETA  = 40    # brightness additive offset

# ---------------------------------------------------------------------------
# Target smoothing — EMA on detected (cx, cy) to reduce jitter
# ---------------------------------------------------------------------------

_TARGET_EMA = 0.35  # weight on new measurement (0=frozen, 1=raw)

# ---------------------------------------------------------------------------
# Annotated stream
# ---------------------------------------------------------------------------

STREAM_PORT = 8082

# ---------------------------------------------------------------------------
# Shared camera state (producer: _camera_thread, consumer: run())
# ---------------------------------------------------------------------------

_frame_lock = threading.Lock()
_latest_frame: np.ndarray | None = None

_jpeg_lock  = threading.Lock()
_latest_jpeg: bytes = b""


# ---------------------------------------------------------------------------
# Camera capture thread (ffmpeg → bgr24)
# ---------------------------------------------------------------------------

def _camera_thread() -> None:
    global _latest_frame
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "v4l2", "-input_format", "yuyv422",
        "-video_size", f"{_CAM_WIDTH}x{_CAM_HEIGHT}",
        "-framerate", str(_CAM_FPS),
        "-i", _CAM_DEVICE,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        print("[cam] ffmpeg started")
        while True:
            raw = proc.stdout.read(_FRAME_BYTES)
            if len(raw) < _FRAME_BYTES:
                print("[cam] stream ended — restarting…")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (_CAM_HEIGHT, _CAM_WIDTH, 3)
            ).copy()
            frame = cv2.convertScaleAbs(frame, alpha=CAM_ALPHA, beta=CAM_BETA)
            with _frame_lock:
                _latest_frame = frame


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_blue(frame: np.ndarray) -> tuple[int, int, int] | None:
    """Return (cx, cy, radius) of the largest blue blob, or None."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt, best_area = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > _MIN_BLOB_AREA and area > best_area:
            best_area = area
            best_cnt  = c

    if best_cnt is None:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    return int(cx), int(cy), int(radius)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _annotate(
    frame: np.ndarray,
    detection: tuple[int, int, int] | None,
    pan_deg: float,
    pitch_pwm: int,
    state: str,
) -> np.ndarray:
    out   = frame.copy()
    fcx   = _CAM_WIDTH  // 2
    fcy   = _CAM_HEIGHT // 2

    # Frame-centre crosshair
    cv2.drawMarker(out, (fcx, fcy), (255, 255, 255),
                   cv2.MARKER_CROSS, markerSize=24, thickness=1)

    if detection:
        tx, ty, r = detection
        cv2.circle(out, (tx, ty), r, (0, 255, 0), 2)
        cv2.drawMarker(out, (tx, ty), (0, 255, 0),
                       cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.line(out, (fcx, fcy), (tx, ty), (0, 200, 0), 1)
        cv2.putText(out, f"({tx},{ty})", (tx + r + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(out, f"[{state}]", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"pan {pan_deg:.1f}deg   pitch {pitch_pwm:+d} pwm",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# MJPEG HTTP server
# ---------------------------------------------------------------------------

class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/":
            html = (
                b"<html><body style='background:#000;margin:0'>"
                b"<img src='/stream' style='width:100%'></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _jpeg_lock:
                        jpeg = _latest_jpeg
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpeg + b"\r\n"
                        )
                    time.sleep(1.0 / _CAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(robot: Robot) -> None:
    global _latest_jpeg

    # Start camera feed
    threading.Thread(target=_camera_thread, daemon=True).start()

    # Start annotated stream server
    server = HTTPServer(("0.0.0.0", STREAM_PORT), _StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ip = socket.gethostbyname(socket.gethostname())
    print(f"[track] stream  → http://{ip}:{STREAM_PORT}")

    # Enable hardware
    robot.enable_servo(PAN_CHANNEL)
    robot.enable_motor(PITCH_MOTOR, DCMotorMode.PWM)

    pan_deg       = PAN_CENTER_DEG
    pitch_pwm     = 0
    scan_dir      = 1
    state         = "SCAN"
    smooth_tx: float | None = None
    smooth_ty: float | None = None
    last_step_at  = 0.0   # monotonic time of last correction step

    robot.set_servo(PAN_CHANNEL, pan_deg)
    robot.set_motor_pwm(PITCH_MOTOR, 0)

    period    = 1.0 / float(DEFAULT_FSM_HZ)
    next_tick = time.monotonic()

    print(f"[track] scanning for #007FFF  pan={pan_deg:.0f}°  Ctrl+C to stop")

    try:
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                time.sleep(0.05)
                continue

            detection = _detect_blue(frame)
            now = time.monotonic()

            if detection:
                tx, ty, _ = detection
                state = "TRACK"

                # EMA smooth detection to reduce frame-to-frame noise
                if smooth_tx is None:
                    smooth_tx, smooth_ty = float(tx), float(ty)
                else:
                    smooth_tx += _TARGET_EMA * (tx - smooth_tx)
                    smooth_ty += _TARGET_EMA * (ty - smooth_ty)

                pan_err   = smooth_tx - (_CAM_WIDTH  // 2)
                pitch_err = smooth_ty - (_CAM_HEIGHT // 2)

                # Pan: step-based (servo holds position between steps)
                if now - last_step_at >= STEP_INTERVAL_S:
                    if abs(pan_err) > CENTER_TOL_PX:
                        pan_step = PAN_STEP_COARSE if abs(pan_err) > COARSE_THRESH_PX else PAN_STEP_FINE
                        pan_deg = float(np.clip(
                            pan_deg - math.copysign(pan_step, pan_err),
                            PAN_MIN_DEG, PAN_MAX_DEG,
                        ))
                        robot.set_servo(PAN_CHANNEL, pan_deg)
                        last_step_at = now

                # Pitch: continuous ±200 toward center, zero when within tolerance
                if abs(pitch_err) > CENTER_TOL_PX:
                    pitch_pwm = int(math.copysign(PITCH_PWM, pitch_err))
                else:
                    pitch_pwm = 0
                robot.set_motor_pwm(PITCH_MOTOR, pitch_pwm)

            else:
                state     = "SCAN"
                smooth_tx = None
                smooth_ty = None
                pitch_pwm = 0
                robot.set_motor_pwm(PITCH_MOTOR, 0)

                # Sweep pan while searching
                if now - last_step_at >= STEP_INTERVAL_S:
                    pan_deg += scan_dir * PAN_SCAN_STEP
                    if pan_deg >= PAN_MAX_DEG:
                        pan_deg  = PAN_MAX_DEG
                        scan_dir = -1
                    elif pan_deg <= PAN_MIN_DEG:
                        pan_deg  = PAN_MIN_DEG
                        scan_dir = 1
                    robot.set_servo(PAN_CHANNEL, pan_deg)
                    last_step_at = now

            # Encode annotated frame for stream
            ann = _annotate(frame, detection, pan_deg, pitch_pwm, state)
            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    finally:
        robot.set_motor_pwm(PITCH_MOTOR, 0)
        robot.disable_motor(PITCH_MOTOR)
        robot.disable_servo(PAN_CHANNEL)
        server.shutdown()
        print("[track] stopped — hardware zeroed")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    class _Node(Node):
        def __init__(self) -> None:
            super().__init__("camera_track")
            self.robot = Robot(self)

    node = _Node()

    def _spin() -> None:
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    def _sighandler(sig, frame):
        raise KeyboardInterrupt()

    old_int  = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    try:
        run(node.robot)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.robot.shutdown()
        except Exception:
            pass
        signal.signal(signal.SIGINT,  old_int)
        signal.signal(signal.SIGTERM, old_term)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
