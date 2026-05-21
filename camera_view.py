"""
camera_view.py — live MJPEG camera viewer
Serves a live feed at http://<pi-ip>:8080 — open in any browser on the same network.
"""

import io
import threading
import time
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

DEVICE   = "/dev/video10"
WIDTH    = 640
HEIGHT   = 480
FPS      = 15
PORT     = 8080

_frame_lock  = threading.Lock()
_latest_jpeg = b""


def _capture_loop() -> None:
    global _latest_jpeg
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {DEVICE} — is pi-camera-feed.service running?")

    print(f"[camera] opened {DEVICE}  {WIDTH}x{HEIGHT} @ {FPS} fps")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera] frame read failed, retrying…")
            time.sleep(0.1)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_index(self):
        html = (
            "<html><body style='background:#000;margin:0'>"
            "<img src='/stream' style='width:100%;height:auto'>"
            "</body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    jpeg = _latest_jpeg
                if jpeg:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    # Wait for first frame
    for _ in range(50):
        with _frame_lock:
            if _latest_jpeg:
                break
        time.sleep(0.1)

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"[viewer] live feed → http://{ip}:{PORT}  (or http://localhost:{PORT})")
    print("[viewer] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] stopped")
