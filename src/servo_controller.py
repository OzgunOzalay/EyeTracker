"""
Servo controller wrapping adafruit-circuitpython-servokit for PCA9685.

Responsibilities:
  - Initialize PCA9685 via ServoKit with explicit I2C bus (required on RPi5)
  - Configure MG996R pulse width range and actuation range
  - Enforce per-channel angle safety limits (hard clamp — never trusts caller)
  - Ramp servos to center on startup and shutdown to avoid mechanical shock
  - Expose apply_delta() as the primary interface for the control loop
  - Expose current_pan / current_tilt for the web status endpoint

RPi5 note: The RPi5 BCM2712 SoC exposes multiple I2C buses. The default GPIO
2/3 header pins map to i2c-1, but ServoKit's implicit default may select a
different bus on RPi5. We pass an explicit busio.I2C object to guarantee
the correct bus is used.
"""

import logging
import time

logger = logging.getLogger(__name__)


class ServoController:
    """PCA9685-backed pan-tilt servo controller."""

    def __init__(self, servo_config: dict) -> None:
        """
        Args:
            servo_config: The 'servo' subtree from config.yaml.

        Raises:
            RuntimeError: If the ServoKit library is missing, or the PCA9685
                          is not found on the I2C bus.
        """
        self._cfg = servo_config

        pan  = servo_config["pan"]
        tilt = servo_config["tilt"]

        self._pan_min    = float(pan["min_angle"])
        self._pan_max    = float(pan["max_angle"])
        self._pan_center = float(pan["center_angle"])
        self._pan_ch     = int(pan["channel"])

        self._tilt_min    = float(tilt["min_angle"])
        self._tilt_max    = float(tilt["max_angle"])
        self._tilt_center = float(tilt["center_angle"])
        self._tilt_ch     = int(tilt["channel"])

        # Publicly readable current angles (used by web status endpoint)
        self.current_pan:  float = self._pan_center
        self.current_tilt: float = self._tilt_center

        self._kit = None
        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        try:
            import board                              # type: ignore[import]
            import busio                              # type: ignore[import]
            from adafruit_servokit import ServoKit    # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "adafruit-circuitpython-servokit or adafruit-blinka is not installed.\n"
                "Run: pip install adafruit-circuitpython-servokit adafruit-blinka"
            ) from exc

        i2c_addr = int(self._cfg.get("i2c_address", 0x40))
        freq     = int(self._cfg.get("frequency", 50))

        # Explicit I2C bus — required on RPi5 where ServoKit's implicit
        # default may select the wrong bus from the multiple available.
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._kit = ServoKit(channels=16, i2c=i2c, address=i2c_addr)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PCA9685 at I2C address 0x{i2c_addr:02X}.\n"
                "Checks:\n"
                "  1. I2C enabled: sudo raspi-config → Interface Options → I2C → Enable\n"
                "  2. HAT seated correctly on GPIO header\n"
                "  3. Address confirmed: i2cdetect -y 1  (should show 40)\n"
                f"Original error: {exc}"
            ) from exc

        # MG996R: 50 Hz PWM, 180-degree actuation range.
        # Pulse width 500–2500 µs covers the full mechanical range.
        # If buzzing occurs at extremes, narrow to 750–2250.
        for ch in (self._pan_ch, self._tilt_ch):
            self._kit.servo[ch].actuation_range = 180
            self._kit.servo[ch].set_pulse_width_range(500, 2500)

        # Ramp to center slowly to avoid mechanical shock on startup
        self._ramp_to_center(steps=30, delay=0.015)

        logger.info(
            "ServoController ready — Pan ch%d [%.0f–%.0f°] center=%.0f°, "
            "Tilt ch%d [%.0f–%.0f°] center=%.0f°",
            self._pan_ch, self._pan_min, self._pan_max, self._pan_center,
            self._tilt_ch, self._tilt_min, self._tilt_max, self._tilt_center,
        )

    def _ramp_to_center(self, steps: int = 30, delay: float = 0.015) -> None:
        """Gradually move both servos to center over (steps × delay) seconds."""
        if self._kit is None:
            return

        # Best-effort read of current position; fall back to center if unavailable
        try:
            start_pan  = self._kit.servo[self._pan_ch].angle  or self._pan_center
            start_tilt = self._kit.servo[self._tilt_ch].angle or self._tilt_center
        except Exception:
            start_pan, start_tilt = self._pan_center, self._tilt_center

        for step in range(1, steps + 1):
            t = step / steps
            self._kit.servo[self._pan_ch].angle  = (
                start_pan  + t * (self._pan_center  - start_pan)
            )
            self._kit.servo[self._tilt_ch].angle = (
                start_tilt + t * (self._tilt_center - start_tilt)
            )
            time.sleep(delay)

        self.current_pan  = self._pan_center
        self.current_tilt = self._tilt_center

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def set_pan(self, angle: float) -> None:
        """Command pan servo to angle (clamped to configured limits)."""
        angle = max(self._pan_min, min(self._pan_max, angle))
        self._kit.servo[self._pan_ch].angle = angle
        self.current_pan = angle

    def set_tilt(self, angle: float) -> None:
        """Command tilt servo to angle (clamped to configured limits)."""
        angle = max(self._tilt_min, min(self._tilt_max, angle))
        self._kit.servo[self._tilt_ch].angle = angle
        self.current_tilt = angle

    def apply_delta(self, pan_delta: float, tilt_delta: float) -> None:
        """
        Increment both axes by their PID-computed deltas.

        This is the primary interface called by the control loop each frame.
        Both axes are clamped independently inside set_pan / set_tilt.
        """
        self.set_pan(self.current_pan   + pan_delta)
        self.set_tilt(self.current_tilt + tilt_delta)

    def center(self) -> None:
        """Immediately jump both servos to center (no ramp)."""
        self.set_pan(self._pan_center)
        self.set_tilt(self._tilt_center)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Ramp to center, then disable PWM output to remove holding torque."""
        if self._kit is None:
            return
        try:
            self._ramp_to_center(steps=30, delay=0.015)
            # Setting angle to None disables the PWM signal, cutting holding
            # torque so the motor doesn't heat up while idle.
            self._kit.servo[self._pan_ch].angle  = None
            self._kit.servo[self._tilt_ch].angle = None
            logger.info("ServoController shut down cleanly.")
        except Exception as exc:
            logger.warning("Error during servo shutdown: %s", exc)
