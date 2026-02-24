"""
Head tracker using OpenCV Haar cascade face detection.

Replaces the MediaPipe FaceMesh backend which fails on Raspberry Pi 5 due
to a known UnicodeDecodeError inside MediaPipe's binary graph loader
(solution_base.py) on ARM64 / Python 3.11.

Responsibilities:
  - Initialize the frontal-face Haar cascade (bundled with OpenCV — no download)
  - Process BGR frames: detect largest face, extract bounding-box centre
  - Compute normalized (error_x, error_y) in [-1.0, 1.0] from frame centre
  - Apply a deadband to suppress micro-jitter when face is centred
  - Draw face rectangle, centre dot, crosshair, and HUD overlay
  - Expose face_absent_duration for the no-face timeout in main.py

Sign convention (matches servo_controller expectations):
  error_x > 0 → face centre is RIGHT of frame centre  → pan servo increases
  error_y > 0 → face centre is BELOW  frame centre     → tilt direction depends on mount
                                                          (configurable via PID kp sign)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result from one frame of head tracking."""
    face_detected: bool
    error_x: float = 0.0          # normalised [-1, 1]; positive = face right of centre
    error_y: float = 0.0          # normalised [-1, 1]; positive = face below centre
    landmark_px: Optional[Tuple[int, int]] = None  # pixel coords of tracked point


class HeadTracker:
    """OpenCV Haar-cascade-based head tracker with overlay rendering."""

    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        """
        Args:
            config:       Full config dict (uses 'tracking' and 'display' subtrees).
            frame_width:  Actual negotiated camera frame width in pixels.
            frame_height: Actual negotiated camera frame height in pixels.
        """
        self._tracking_cfg = config["tracking"]
        self._display_cfg  = config.get("display", {})
        self._frame_w = frame_width
        self._frame_h = frame_height

        self._deadband        = float(self._tracking_cfg.get("deadband", 0.03))
        self._no_face_timeout = float(self._tracking_cfg.get("no_face_timeout", 1.5))
        # Scale factor for the detection image: 0.25 → 480×270 at 1080p, ~16× fewer
        # pixels for detectMultiScale, keeping FPS high without hurting accuracy.
        self._detection_scale = float(self._tracking_cfg.get("detection_scale", 0.25))

        self._last_face_time: float = time.monotonic()
        self._face_absent_logged: bool = False

        # OpenCV Haar cascade — ships with every OpenCV install, no download needed
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}. "
                "Reinstall opencv-python."
            )

        logger.info(
            "HeadTracker ready (OpenCV Haar cascade). "
            "deadband=%.3f, detection_scale=%.2f (%dx%d detect), frame=%dx%d",
            self._deadband, self._detection_scale,
            int(frame_width * self._detection_scale),
            int(frame_height * self._detection_scale),
            frame_width, frame_height,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def face_absent_duration(self) -> float:
        """Seconds elapsed since the last frame with a detected face."""
        return time.monotonic() - self._last_face_time

    def process_frame(
        self,
        frame: np.ndarray,
        pan_angle: float,
        tilt_angle: float,
    ) -> TrackingResult:
        """
        Run face detection on one BGR frame, draw overlay in-place, and return
        the tracking error signal.

        Args:
            frame:      BGR frame from cv2.VideoCapture (modified in-place).
            pan_angle:  Current pan servo angle for HUD display.
            tilt_angle: Current tilt servo angle for HUD display.

        Returns:
            TrackingResult with face_detected, error_x, error_y, landmark_px.
        """
        frame_h, frame_w = frame.shape[:2]
        cx = frame_w / 2.0
        cy = frame_h / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downsample for detection — Haar cascade is O(pixels), so 0.25× scale
        # reduces the detection image from ~2 MP to ~130 K pixels (~16× faster).
        # Bounding boxes are scaled back to full resolution after detection.
        scale  = self._detection_scale
        det_w  = max(1, int(frame_w * scale))
        det_h  = max(1, int(frame_h * scale))
        small  = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_LINEAR)

        # minSize in the *downscaled* image — keep ~4 % of scaled height so
        # the same real-world minimum face size is enforced regardless of scale.
        min_dim_small = max(8, int(det_h * 0.04))
        raw_faces = self._detector.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_dim_small, min_dim_small),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Scale bounding boxes back to full-resolution coordinates
        if len(raw_faces) > 0:
            inv = 1.0 / scale
            faces = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                     for x, y, w, h in raw_faces]
        else:
            faces = []

        result = TrackingResult(face_detected=False)

        if faces:
            # Pick the largest detected face by area
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_cx = x + w // 2
            face_cy = y + h // 2

            # Draw bounding box and centre dot
            lm_color = tuple(
                int(c) for c in self._display_cfg.get("landmark_color", [0, 200, 255])
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), lm_color, 2)
            cv2.circle(frame, (face_cx, face_cy), 6, lm_color, -1)
            cv2.circle(frame, (face_cx, face_cy), 8, (0, 0, 0), 1)

            # Normalised error: [-1, 1] relative to half-frame dimension
            raw_ex = (face_cx - cx) / (frame_w / 2.0)
            raw_ey = (face_cy - cy) / (frame_h / 2.0)

            # Deadband — zero out small errors to suppress servo jitter
            error_x = raw_ex if abs(raw_ex) > self._deadband else 0.0
            error_y = raw_ey if abs(raw_ey) > self._deadband else 0.0

            result = TrackingResult(
                face_detected=True,
                error_x=error_x,
                error_y=error_y,
                landmark_px=(face_cx, face_cy),
            )

            self._last_face_time = time.monotonic()
            self._face_absent_logged = False

        else:
            if not self._face_absent_logged:
                logger.debug("No face detected in frame.")
                self._face_absent_logged = True

        # Overlay elements drawn regardless of detection status
        self._draw_crosshair(frame, cx, cy)
        self._draw_hud(frame, result, pan_angle, tilt_angle)

        return result

    def close(self) -> None:
        """No-op — CascadeClassifier holds no external resources."""

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_crosshair(self, frame: np.ndarray, cx: float, cy: float) -> None:
        color     = tuple(
            int(c) for c in self._display_cfg.get("crosshair_color", [0, 255, 0])
        )
        thickness = int(self._display_cfg.get("line_thickness", 1))
        h, w      = frame.shape[:2]
        icx, icy  = int(cx), int(cy)
        gap = 16   # pixel gap around the centre reticle

        cv2.line(frame, (0,         icy), (icx - gap, icy), color, thickness)
        cv2.line(frame, (icx + gap, icy), (w - 1,     icy), color, thickness)
        cv2.line(frame, (icx, 0),         (icx, icy - gap), color, thickness)
        cv2.line(frame, (icx, icy + gap), (icx, h - 1),     color, thickness)
        cv2.circle(frame, (icx, icy), gap // 2, color, thickness)

    def _draw_hud(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        pan_angle: float,
        tilt_angle: float,
    ) -> None:
        color  = tuple(
            int(c) for c in self._display_cfg.get("text_color", [255, 255, 255])
        )
        scale  = float(self._display_cfg.get("font_scale", 0.55))
        font   = cv2.FONT_HERSHEY_SIMPLEX
        h, w   = frame.shape[:2]
        line_h = int(scale * 32)
        pad    = 6

        lines = [
            f"Pan:  {pan_angle:6.1f} deg",
            f"Tilt: {tilt_angle:6.1f} deg",
        ]

        if result.face_detected:
            lines += [
                f"Err X: {result.error_x:+.3f}",
                f"Err Y: {result.error_y:+.3f}",
            ]
            status_color = (0, 220, 0)
            status_text  = "TRACKING"
        else:
            absent = self.face_absent_duration
            lines.append(f"No face  ({absent:.1f}s)")
            status_color = (0, 80, 255)
            status_text  = "SEARCHING"

        # Semi-transparent background rectangle
        rect_w = 240
        rect_h = len(lines) * line_h + 2 * pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h
            cv2.putText(frame, line, (pad, y), font, scale, color, 1, cv2.LINE_AA)

        # Status label at bottom-left
        cv2.putText(
            frame, status_text,
            (pad, h - pad - 4),
            font, scale * 0.9, status_color, 1, cv2.LINE_AA,
        )
