# EyeTracker — Phase 1: Pan-Tilt Head Tracking

Raspberry Pi 5 system that tracks a participant's head using MediaPipe FaceMesh
and drives a PCA9685-controlled MG996R servo pan-tilt mount to keep the camera
aimed at the face.

## Hardware

| Component         | Details                              |
|-------------------|--------------------------------------|
| SBC               | Raspberry Pi 5                       |
| Tracking camera   | Arducam B0498 on `/dev/video0`       |
| Servo driver      | PCA9685 HAT, I2C address `0x40`      |
| Pan servo         | MG996R on PCA9685 channel 0          |
| Tilt servo        | MG996R on PCA9685 channel 1          |

## Project Structure

```
EyeTracker/
├── main.py                  # Entry point — control loop
├── config.yaml              # All tunable parameters
├── requirements.txt         # Python dependencies
├── src/
│   ├── camera.py            # OpenCV camera wrapper (V4L2 / MJPEG)
│   ├── head_tracker.py      # MediaPipe FaceMesh detection + overlay
│   ├── pid.py               # Discrete PID with anti-windup
│   ├── servo_controller.py  # PCA9685 ServoKit wrapper
│   └── web_server.py        # Flask MJPEG stream + JSON status API
└── templates/
    └── index.html           # Live web dashboard
```

## Quick Start

### 1. System prerequisites

```bash
sudo apt update && sudo apt install -y \
    python3-dev python3-pip python3-venv \
    libatlas-base-dev libhdf5-dev libjpeg-dev \
    v4l-utils i2c-tools

# Enable I2C
sudo raspi-config nonint do_i2c 0
sudo usermod -aG i2c $USER
newgrp i2c

# Verify camera is visible
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Verify PCA9685 is on I2C bus (should show 40 in the grid)
i2cdetect -y 1
```

### 2. Python environment

```bash
cd ~/EyeTracker
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run

```bash
python3 main.py
```

- **Local display:** An OpenCV window opens showing the annotated camera feed.
  Press **q** to quit cleanly.
- **Web dashboard:** Open `http://<raspberry-pi-ip>:5000` in any browser on
  your LAN for the live stream, servo angle gauges, and tracking status.
- **Headless / SSH:** Set `display.show_window: false` in `config.yaml`.
  The system runs without an X display; use the web UI only.

## Configuration (`config.yaml`)

All parameters are documented inline in `config.yaml`.  Key ones to tune:

| Key | Default | Effect |
|-----|---------|--------|
| `pid.pan.kp` | 25.0 | Aggressiveness of horizontal tracking |
| `pid.pan.kd` | 3.5 | Damping for fast head turns |
| `pid.tilt.kp` | 20.0 | Aggressiveness of vertical tracking |
| `tracking.deadband` | 0.03 | Error threshold below which servos don't move (prevents jitter) |
| `tracking.no_face_timeout` | 1.5 s | After this long without a face, reset PID integrators |
| `servo.pan.min/max_angle` | 30–150° | Hard safety limits for pan servo |
| `servo.tilt.min/max_angle` | 45–135° | Hard safety limits for tilt servo |
| `web.stream_width/height` | 960×540 | Resolution of MJPEG stream (downscaled from 1920×1080) |
| `display.show_window` | true | Set false for headless operation |

## PID Tuning

### Procedure

1. **Proportional only** — set `ki: 0`, `kd: 0`. Increase `kp` until the
   servo oscillates at rest with a face centered. Back off to ~70% of that value.

2. **Add derivative** — start `kd` at `kp × 0.15`. Oscillation should stop.
   Too much `kd` makes tracking sluggish on fast movements.

3. **Add integral** — start `ki` at `kp × 0.01`. Corrects small steady-state
   offset where the servo holds the face slightly off-center. Too much causes
   slow sinusoidal drift.

4. **Tune deadband** — if the servo buzzes when face is centered, increase
   `tracking.deadband` from 0.03 toward 0.05.

### Tilt direction inversion

If tilting your head up causes the camera to tilt **down** (wrong direction),
set `pid.tilt.kp` to a **negative** value in `config.yaml`. No code change needed.

## Web Dashboard

Open `http://<pi-ip>:5000` in a browser:

- **Live stream** — annotated MJPEG feed at 960×540
- **Servo gauges** — visual pan and tilt angle bars
- **Error bars** — normalized X/Y tracking error (-1 to +1)
- **FPS counter** — control loop frame rate
- **Status indicator** — TRACKING (green) / SEARCHING (orange)

The dashboard polls `/status` every 200 ms and reconnects automatically if
the stream drops.

## Servo Wiring

| Servo | PCA9685 Channel | Function |
|-------|-----------------|----------|
| MG996R #1 | 0 | Pan (horizontal rotation) |
| MG996R #2 | 1 | Tilt (vertical rotation)  |

Both servos start at 90° (center) on launch and return to center on shutdown.

## Phase 2 (planned)

Mount the Arducam IR Mirai Mono on the servo pan-tilt base.
Use the pan-tilt position found in Phase 1 (head tracking) to aim the IR
camera at the left eye, then switch to fine-grained pupil tracking.
