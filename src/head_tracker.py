"""
Head tracker using MediaPipe FaceMesh.

Responsibilities:
  - Initialize the FaceMesh detector (max 1 face, refine_landmarks=True)
  - Process BGR frames: convert to RGB, run inference, extract nose tip landmark
  - Compute normalized (error_x, error_y) in [-1.0, 1.0] from frame center
  - Apply a deadband to suppress micro-jitter when face is centered
  - Draw face mesh tessellation, contours, nose-tip dot, crosshair, and HUD overlay
  - Expose face_absent_duration for the no-face timeout in main.py

Sign convention (matches servo_controller expectations):
  error_x > 0 → nose is RIGHT of center  → pan servo should increase angle
  error_y > 0 → nose is BELOW center     → tilt servo direction depends on mount
                                            (configurable via PID kp sign in config.yaml)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result from one frame of head tracking."""
    face_detected: bool
    error_x: float = 0.0          # normalized [-1, 1]; positive = nose right of center
    error_y: float = 0.0          # normalized [-1, 1]; positive = nose below center
    landmark_px: Optional[Tuple[int, int]] = None  # pixel coords of tracked landmark


class HeadTracker:
    """MediaPipe FaceMesh-based head tracker with overlay rendering."""

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

        self._landmark_index  = int(self._tracking_cfg.get("landmark_index", 4))
        self._deadband        = float(self._tracking_cfg.get("deadband", 0.03))
        self._no_face_timeout = float(self._tracking_cfg.get("no_face_timeout", 1.5))

        self._last_face_time: float = time.monotonic()
        self._face_absent_logged: bool = False

        # MediaPipe FaceMesh setup
        mp_fm = mp.solutions.face_mesh
        self._face_mesh = mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,           # enables iris landmarks (468+)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_drawing        = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_face_mesh      = mp_fm

        logger.info(
            "HeadTracker ready. Landmark index=%d, deadband=%.3f, "
            "frame=%dx%d",
            self._landmark_index, self._deadband, frame_width, frame_height,
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

        # MediaPipe requires RGB; mark non-writeable to avoid an internal copy
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        result = TrackingResult(face_detected=False)

        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]

            # Draw tessellation and contours
            self._mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    self._mp_drawing_styles.get_default_face_mesh_tesselation_style()
                ),
            )
            self._mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    self._mp_drawing_styles.get_default_face_mesh_contours_style()
                ),
            )

            # Extract nose tip landmark (index 4 by default)
            lm = face_landmarks.landmark[self._landmark_index]
            nose_px = (int(lm.x * frame_w), int(lm.y * frame_h))

            # Draw tracked point
            lm_color = tuple(
                int(c) for c in self._display_cfg.get("landmark_color", [0, 200, 255])
            )
            cv2.circle(frame, nose_px, 7, lm_color, -1)
            cv2.circle(frame, nose_px, 9, (0, 0, 0), 1)

            # Normalized error: range [-1, 1] relative to half-frame dimension.
            # Resolution-independent: gains in config.yaml don't need to change
            # if camera resolution changes.
            raw_ex = (nose_px[0] - cx) / (frame_w / 2.0)
            raw_ey = (nose_px[1] - cy) / (frame_h / 2.0)

            # Apply deadband — zero out small errors to suppress servo jitter
            error_x = raw_ex if abs(raw_ex) > self._deadband else 0.0
            error_y = raw_ey if abs(raw_ey) > self._deadband else 0.0

            result = TrackingResult(
                face_detected=True,
                error_x=error_x,
                error_y=error_y,
                landmark_px=nose_px,
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
        """Release MediaPipe resources."""
        self._face_mesh.close()

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
        gap = 16   # pixel gap around the center reticle

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
        color     = tuple(
            int(c) for c in self._display_cfg.get("text_color", [255, 255, 255])
        )
        scale     = float(self._display_cfg.get("font_scale", 0.55))
        font      = cv2.FONT_HERSHEY_SIMPLEX
        h, w      = frame.shape[:2]
        line_h    = int(scale * 32)
        pad       = 6

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
