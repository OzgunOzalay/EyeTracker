"""
Flask-based web server for real-time monitoring of the EyeTracker.

Runs in a background daemon thread started by main.py.  The control loop
writes state to a SharedState object; the Flask routes read from it.

Routes:
  GET /          → Serves templates/index.html (the live dashboard)
  GET /stream    → MJPEG multipart stream of the annotated camera feed
  GET /status    → JSON snapshot of current tracking state

Threading model:
  - SharedState uses a threading.Lock for all reads and writes.
  - The MJPEG generator in /stream blocks on threading.Event, woken up each
    time the control loop deposits a new frame.  This avoids busy-polling and
    limits stream FPS to actual camera FPS without extra sleep() calls.

Frame handling:
  - Frames are downscaled to stream_width × stream_height before JPEG encoding
    to keep network bandwidth practical at 1920×1080 source resolution.
  - The encoded JPEG bytes are stored in SharedState so multiple browser
    clients receive the same pre-encoded bytes without re-encoding per client.
"""

import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state between the control loop and the Flask server
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    """Thread-safe container for tracking data shared between threads."""
    pan_angle:    float = 90.0
    tilt_angle:   float = 90.0
    face_detected: bool = False
    error_x:      float = 0.0
    error_y:      float = 0.0
    fps:          float = 0.0

    _lock:        threading.Lock  = field(default_factory=threading.Lock, repr=False)
    _frame_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _jpeg_bytes:  Optional[bytes] = field(default=None, repr=False)

    def update(
        self,
        frame: np.ndarray,
        pan_angle: float,
        tilt_angle: float,
        face_detected: bool,
        error_x: float,
        error_y: float,
        fps: float,
        stream_width: int,
        stream_height: int,
        stream_quality: int,
    ) -> None:
        """
        Called by the control loop each frame.  Encodes the frame to JPEG once
        so all /stream clients share the same bytes.
        """
        # Downscale for streaming
        small = cv2.resize(frame, (stream_width, stream_height),
                           interpolation=cv2.INTER_LINEAR)
        ret, buf = cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, stream_quality]
        )
        jpeg = buf.tobytes() if ret else None

        with self._lock:
            self.pan_angle    = pan_angle
            self.tilt_angle   = tilt_angle
            self.face_detected = face_detected
            self.error_x      = error_x
            self.error_y      = error_y
            self.fps          = fps
            self._jpeg_bytes  = jpeg

        self._frame_event.set()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "pan":          round(self.pan_angle, 1),
                "tilt":         round(self.tilt_angle, 1),
                "face_detected": self.face_detected,
                "error_x":      round(self.error_x, 4),
                "error_y":      round(self.error_y, 4),
                "fps":          round(self.fps, 1),
            }

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg_bytes

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """Block until a new frame is available or timeout expires."""
        got = self._frame_event.wait(timeout=timeout)
        self._frame_event.clear()
        return got


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(shared_state: SharedState) -> Flask:
    app = Flask(__name__, template_folder="../templates")
    # Suppress Flask's default request logging to keep the console clean
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/stream")
    def stream():
        """MJPEG multipart stream."""
        def generate():
            while True:
                shared_state.wait_for_frame(timeout=1.0)
                jpeg = shared_state.get_jpeg()
                if jpeg is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status")
    def status():
        return jsonify(shared_state.get_status())

    return app


# ---------------------------------------------------------------------------
# Background thread launcher
# ---------------------------------------------------------------------------

def start_web_server(shared_state: SharedState, web_config: dict) -> threading.Thread:
    """
    Start Flask in a background daemon thread.

    Args:
        shared_state: The SharedState instance to read from.
        web_config:   The 'web' subtree from config.yaml.

    Returns:
        The started Thread object (daemon=True; stops automatically when
        the main process exits).
    """
    host = web_config.get("host", "0.0.0.0")
    port = int(web_config.get("port", 5000))

    app = create_app(shared_state)

    thread = threading.Thread(
        target=lambda: app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,   # must be False in a non-main thread
            threaded=True,
        ),
        daemon=True,
        name="WebServer",
    )
    thread.start()
    logger.info("Web server started at http://%s:%d", host, port)
    return thread
