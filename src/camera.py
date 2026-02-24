"""
Camera abstraction wrapping cv2.VideoCapture.

Provides a context-manager and generator interface:

    with CameraCapture(config["camera"]) as cam:
        for frame in cam.frames():
            process(frame)

Design notes:
  - Uses CAP_V4L2 backend explicitly — required on Raspberry Pi / Linux for
    reliable frame delivery at the requested FPS.
  - Requests MJPG codec via CAP_PROP_FOURCC so the Arducam B0498 streams
    compressed frames over USB rather than raw YUYV, which would saturate
    the bus at 1920×1080.
  - Frame read failures yield None rather than raising, so the caller can
    decide to skip and continue rather than crashing the control loop.
"""

import logging
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """OpenCV VideoCapture wrapper with context-manager and generator interface."""

    def __init__(self, camera_config: dict) -> None:
        """
        Args:
            camera_config: The 'camera' subtree from config.yaml.

        Raises:
            RuntimeError: If the camera device cannot be opened (raised inside
                          __enter__ / _open, not in __init__).
        """
        self._cfg    = camera_config
        self._cap    = None
        self._index  = int(camera_config.get("device_index", 0))
        self._width  = int(camera_config.get("width",  1920))
        self._height = int(camera_config.get("height", 1080))
        self._fps    = int(camera_config.get("fps",    60))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self._open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Generator interface
    # ------------------------------------------------------------------

    def frames(self) -> Generator[np.ndarray | None, None, None]:
        """
        Yield BGR frames indefinitely.  Yields None on a failed read so the
        caller can log and continue rather than catching an exception.
        """
        while True:
            if self._cap is None or not self._cap.isOpened():
                yield None
                return
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Camera read failed (ret=%s); yielding None.", ret)
                yield None
            else:
                yield frame

    # ------------------------------------------------------------------
    # Properties (reflect actual negotiated values, not requested ones)
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    @property
    def actual_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        return float(self._fps)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the camera device and configure resolution/codec."""
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at /dev/video{self._index}. "
                "Check that the camera is connected and not in use by another process.\n"
                "Diagnostic: run  v4l2-ctl --list-devices"
            )

        # Request MJPG so the Arducam B0498 streams compressed frames over USB.
        # Without this the driver defaults to YUYV which saturates USB bandwidth
        # at 1920x1080 and causes severe frame drops.
        self._cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

        actual_w   = self.width
        actual_h   = self.height
        actual_fps = self.actual_fps

        logger.info(
            "Camera /dev/video%d opened: %dx%d @ %.1f fps",
            self._index, actual_w, actual_h, actual_fps,
        )

        if actual_w != self._width or actual_h != self._height:
            logger.warning(
                "Requested %dx%d but driver negotiated %dx%d. "
                "Check supported modes with: v4l2-ctl --device=/dev/video%d --list-formats-ext",
                self._width, self._height, actual_w, actual_h, self._index,
            )

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")
