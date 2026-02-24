"""
Discrete PID controller with anti-windup and output clamping.

The controller operates on a normalized error signal in [-1.0, 1.0] and
produces an output delta in degrees that the servo controller adds to the
current servo angle.  All tunable parameters are injected at construction
time from config.yaml so no magic constants live inside this module.

Design decisions:
  - Derivative-on-measurement (not on error) to avoid derivative kick on
    sudden re-acquisition after a face-absence timeout.
  - dt measured from wall-clock time (time.monotonic) rather than assumed
    fixed, because frame delivery from the camera varies with CPU load.
  - Anti-windup: integral accumulator clamped to ±integral_limit before
    multiplying by ki, bounding the maximum integral contribution.
"""

import time


class PIDController:
    """Discrete PID with anti-windup, output clamping, and derivative-on-measurement."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_output: float,
        integral_limit: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_measurement: float = 0.0
        self._last_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, error: float, measurement: float) -> float:
        """
        Compute one PID step and return the output (delta degrees).

        Args:
            error:       Set-point minus measurement (set-point is always 0
                         — the face should be at frame center).
            measurement: Raw measurement value (normalized nose position).
                         Used for derivative-on-measurement to avoid spikes
                         on sudden face reappearance.

        Returns:
            Output delta in degrees, clamped to ±max_output.
            Returns 0.0 on the very first call (no dt available yet).
        """
        now = time.monotonic()

        if self._last_time is None:
            self._last_time = now
            self._prev_measurement = measurement
            return 0.0

        dt = now - self._last_time
        self._last_time = now

        # Guard against near-zero dt (two calls within the same millisecond)
        if dt < 1e-6:
            dt = 1e-6

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral),
        )
        i_term = self.ki * self._integral

        # Derivative-on-measurement: negate the measurement derivative so
        # increasing measurement (moving away from center) produces a braking term.
        d_measurement = (measurement - self._prev_measurement) / dt
        d_term = -self.kd * d_measurement
        self._prev_measurement = measurement

        output = p_term + i_term + d_term
        return max(-self.max_output, min(self.max_output, output))

    def reset(self) -> None:
        """
        Reset integrator and derivative state.

        Call this when face detection is re-acquired after a timeout, or on
        shutdown.  Does NOT reset kp/ki/kd gains.
        """
        self._integral = 0.0
        self._prev_measurement = 0.0
        self._last_time = None
