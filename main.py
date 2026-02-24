"""
EyeTracker Phase 1 — Raspberry Pi 5 pan-tilt head tracking.

Entry point.  Wires up all components and runs the main control loop.

Control loop (per frame):
  1. Read frame from camera (CameraCapture)
  2. Run FaceMesh face detection (HeadTracker) → normalized error signal
  3. Feed error to per-axis PID controllers → angle deltas (degrees)
  4. Apply deltas to servos via ServoController
  5. Publish annotated frame + state to SharedState for the web server
  6. Optionally display annotated frame in local OpenCV window

No-face handling:
  - While no face is detected, servos hold the last commanded position.
  - After no_face_timeout seconds of absence, PID integrators are reset once
    so stale integral error doesn't cause a lurch on re-acquisition.

Shutdown:
  - Ctrl-C or SIGTERM → sets running=False, exits loop cleanly.
  - Press 'q' in the OpenCV window for the same effect.
  - On exit: servos ramp to center → PWM disabled, camera released.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.camera          import CameraCapture
from src.head_tracker    import HeadTracker
from src.pid             import PIDController
from src.servo_controller import ServoController
from src.web_server      import SharedState, start_web_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Ensure config.yaml is in the project root directory."
        )
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_pid(pid_cfg: dict) -> PIDController:
    return PIDController(
        kp=float(pid_cfg["kp"]),
        ki=float(pid_cfg["ki"]),
        kd=float(pid_cfg["kd"]),
        max_output=float(pid_cfg["max_output"]),
        integral_limit=float(pid_cfg["integral_limit"]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    setup_logging()
    log = logging.getLogger("main")

    # ── Config ─────────────────────────────────────────────────────────────
    try:
        config = load_config("config.yaml")
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    display_cfg = config.get("display", {})
    show_window = display_cfg.get("show_window", True)
    window_title = display_cfg.get("window_title", "EyeTracker Phase 1")

    web_cfg = config.get("web", {})
    web_enabled    = web_cfg.get("enabled", True)
    stream_width   = int(web_cfg.get("stream_width",  960))
    stream_height  = int(web_cfg.get("stream_height", 540))
    stream_quality = int(web_cfg.get("stream_quality", 80))

    tracking_cfg     = config["tracking"]
    no_face_timeout  = float(tracking_cfg.get("no_face_timeout", 1.5))

    # ── Shared state for web server ─────────────────────────────────────────
    shared_state = SharedState()

    # ── Web server ──────────────────────────────────────────────────────────
    if web_enabled:
        start_web_server(shared_state, web_cfg)

    # ── Servos ─────────────────────────────────────────────────────────────
    servo: Optional[ServoController] = None
    try:
        servo = ServoController(config["servo"])
    except RuntimeError as exc:
        log.error("Servo initialization failed:\n%s", exc)
        log.warning(
            "Continuing in DISPLAY-ONLY mode — no servo movement. "
            "Fix the I2C/HAT issue and restart to enable tracking."
        )

    # ── PID controllers ─────────────────────────────────────────────────────
    pid_pan  = build_pid(config["pid"]["pan"])
    pid_tilt = build_pid(config["pid"]["tilt"])

    # ── Graceful shutdown ────────────────────────────────────────────────────
    running = True
    pid_integrators_reset = False   # tracks whether we've reset PIDs this absence

    def _stop(signum, _frame):
        nonlocal running
        log.info("Signal %d received; shutting down.", signum)
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── FPS tracking ─────────────────────────────────────────────────────────
    frame_count = 0
    fps_timer   = time.monotonic()
    display_fps = 0.0

    exit_code = 0

    # ── Camera + control loop ─────────────────────────────────────────────────
    try:
        with CameraCapture(config["camera"]) as cam:
            tracker = HeadTracker(config, cam.width, cam.height)
            log.info(
                "Control loop running. Camera: %dx%d. "
                "%s",
                cam.width, cam.height,
                f"Web UI: http://0.0.0.0:{web_cfg.get('port', 5000)}"
                if web_enabled else "Web UI: disabled",
            )

            for frame in cam.frames():
                if not running:
                    break

                if frame is None:
                    time.sleep(0.05)
                    continue

                # Current servo angles for HUD
                pan_angle  = servo.current_pan  if servo else 90.0
                tilt_angle = servo.current_tilt if servo else 90.0

                # ── Face detection ──────────────────────────────────────────
                result = tracker.process_frame(frame, pan_angle, tilt_angle)

                # ── Control ─────────────────────────────────────────────────
                if result.face_detected:
                    # Reset flag so integrators get reset again next absence
                    pid_integrators_reset = False

                    pan_delta  = pid_pan.compute( result.error_x, result.error_x)
                    # Negate error_y: positive error_y = face below center.
                    # Whether that means tilt should increase or decrease depends
                    # on bracket orientation. Adjust kp sign in config.yaml if wrong.
                    tilt_delta = pid_tilt.compute(result.error_y, result.error_y)

                    if servo:
                        servo.apply_delta(pan_delta, tilt_delta)

                else:
                    # No face: hold position, reset integrators once after timeout
                    if (
                        tracker.face_absent_duration > no_face_timeout
                        and not pid_integrators_reset
                    ):
                        pid_pan.reset()
                        pid_tilt.reset()
                        pid_integrators_reset = True
                        log.debug(
                            "PID integrators reset after %.1fs without face.",
                            tracker.face_absent_duration,
                        )

                # ── FPS counter ─────────────────────────────────────────────
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.monotonic()
                    display_fps = 30.0 / max(now - fps_timer, 1e-6)
                    fps_timer = now

                # ── Publish to web server ────────────────────────────────────
                if web_enabled:
                    shared_state.update(
                        frame          = frame,
                        pan_angle      = servo.current_pan  if servo else 90.0,
                        tilt_angle     = servo.current_tilt if servo else 90.0,
                        face_detected  = result.face_detected,
                        error_x        = result.error_x,
                        error_y        = result.error_y,
                        fps            = display_fps,
                        stream_width   = stream_width,
                        stream_height  = stream_height,
                        stream_quality = stream_quality,
                    )

                # ── Local display ─────────────────────────────────────────────
                if show_window:
                    # FPS stamp top-right
                    h, w = frame.shape[:2]
                    cv2.putText(
                        frame,
                        f"FPS: {display_fps:.1f}",
                        (w - 130, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1, cv2.LINE_AA,
                    )
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        log.info("'q' pressed — exiting.")
                        running = False

    except RuntimeError as exc:
        log.error("Fatal error: %s", exc)
        exit_code = 1
    except Exception as exc:
        log.exception("Unexpected error in control loop: %s", exc)
        exit_code = 1
    finally:
        log.info("Shutting down…")
        if show_window:
            cv2.destroyAllWindows()
        if servo:
            servo.shutdown()
        log.info("Done.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
