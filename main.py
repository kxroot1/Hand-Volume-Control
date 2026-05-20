"""
Hand Gesture Volume Control for Windows
========================================
Controls system master volume using hand gestures detected via webcam.

Requirements:
    pip install opencv-python mediapipe pycaw numpy comtypes

Usage:
    python main.py
    Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import tempfile
import urllib.request

# ──────────────────────────────────────────────
# 1. AUDIO SETUP
# ──────────────────────────────────────────────

def setup_audio():
    """
    Initialise pycaw with a 3-method fallback chain that works across all
    pycaw versions.
    Returns (volume_interface, vol_range) or raises RuntimeError.
    """
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        # Method 1: newer pycaw returns a COM IMMDevice pointer directly
        try:
            speakers  = AudioUtilities.GetSpeakers()
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume    = cast(interface, POINTER(IAudioEndpointVolume))
            print("[Audio] Initialised via IMMDevice.Activate (new pycaw).")
            return volume, (0.0, 1.0)
        except Exception:
            pass

        # Method 2: comtypes IMMDeviceEnumerator (version-independent)
        try:
            import comtypes
            from pycaw.pycaw import CLSID_MMDeviceEnumerator, IMMDeviceEnumerator
            enumerator = comtypes.CoCreateInstance(
                CLSID_MMDeviceEnumerator,
                IMMDeviceEnumerator,
                comtypes.CLSCTX_INPROC_SERVER,
            )
            default_device = enumerator.GetDefaultAudioEndpoint(0, 0)
            interface = default_device.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            print("[Audio] Initialised via IMMDeviceEnumerator (method 2).")
            return volume, (0.0, 1.0)
        except Exception:
            pass

        # Method 3: legacy pycaw wraps the COM pointer in ._dev
        speakers  = AudioUtilities.GetSpeakers()
        raw_dev   = getattr(speakers, "_dev", speakers)
        interface = raw_dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume    = cast(interface, POINTER(IAudioEndpointVolume))
        print("[Audio] Initialised via legacy AudioDevice._dev (method 3).")
        return volume, (0.0, 1.0)

    except Exception as exc:
        raise RuntimeError(f"[Audio] Failed to initialise pycaw: {exc}") from exc


def set_volume(volume_interface, level_0_to_100: float):
    """Set master volume; level is 0-100 (float)."""
    scalar = float(np.clip(level_0_to_100 / 100.0, 0.0, 1.0))
    volume_interface.SetMasterVolumeLevelScalar(scalar, None)


def get_volume(volume_interface) -> float:
    return volume_interface.GetMasterVolumeLevelScalar() * 100.0


def mute_audio(volume_interface, mute: bool):
    volume_interface.SetMute(int(mute), None)


def is_muted(volume_interface) -> bool:
    return bool(volume_interface.GetMute())


# ──────────────────────────────────────────────
# 2. CAMERA SETUP
# ──────────────────────────────────────────────

def setup_camera() -> cv2.VideoCapture:
    """
    Try camera indices 0-2 with CAP_DSHOW first, then CAP_ANY as fallback.
    Does NOT force a resolution - uses whatever the camera natively reports
    to avoid the cv2.Mat step-mismatch assertion error on some Windows drivers.
    Returns an open VideoCapture or raises RuntimeError.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    for idx in range(3):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                # Drain a frame to confirm the stream is truly live
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    cap.release()
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w <= 0 or h <= 0:
                    h, w = frame.shape[:2]  # fall back to actual frame shape
                print(f"[Camera] Opened index {idx}, backend {backend}, native {w}x{h}")
                return cap
            except Exception as exc:
                print(f"[Camera] Index {idx} backend {backend} failed: {exc}")
                try:
                    cap.release()
                except Exception:
                    pass
    raise RuntimeError("[Camera] No accessible camera found on indices 0-2.")


# ──────────────────────────────────────────────
# 3. MEDIAPIPE SETUP
#    Tries new Tasks API (>=0.10) first, then legacy solutions API
# ──────────────────────────────────────────────

def setup_mediapipe():
    """
    Returns:
        detector      - hands detector object
        use_tasks     - True = Tasks API, False = legacy solutions
        drawing_stuff - (mp_drawing, mp_draw_styles, HAND_CONNECTIONS) or None
    """
    e_tasks = None

    # ── Try new Tasks API (mediapipe >= 0.10) ─────────────────────────────
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )

        MODEL_PATH = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        MODEL_URL  = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

        if not os.path.exists(MODEL_PATH):
            print("[MediaPipe] Downloading hand_landmarker.task (~6 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[MediaPipe] Download complete.")

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        detector = HandLandmarker.create_from_options(options)
        print("[MediaPipe] Using new Tasks API (mediapipe >= 0.10).")
        return detector, True, None

    except Exception as exc:
        e_tasks = exc
        print(f"[MediaPipe] Tasks API unavailable ({exc}), trying legacy...")

    # ── Fall back to legacy solutions API ─────────────────────────────────
    try:
        mp_hands_mod   = mp.solutions.hands
        mp_drawing     = mp.solutions.drawing_utils
        mp_draw_styles = mp.solutions.drawing_styles
        detector = mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        drawing_stuff = (mp_drawing, mp_draw_styles, mp_hands_mod.HAND_CONNECTIONS)
        print("[MediaPipe] Using legacy solutions API.")
        return detector, False, drawing_stuff

    except Exception as e_leg:
        raise RuntimeError(
            f"[MediaPipe] Both APIs failed.\n  Tasks: {e_tasks}\n  Legacy: {e_leg}"
        )


def detect_hands(detector, frame_rgb, use_tasks):
    """
    Run hand detection on an RGB numpy frame.
    Returns list of 21 landmarks (each with .x .y .z) for the first hand,
    or None if no hand is detected.
    """
    if use_tasks:
        import mediapipe as _mp
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]
    else:
        result = detector.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark


# ──────────────────────────────────────────────
# 4. HELPER UTILITIES
# ──────────────────────────────────────────────

def get_landmark_px(lm, w: int, h: int) -> tuple:
    return int(lm.x * w), int(lm.y * h)


def euclidean_distance(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class Smoother:
    """Exponential moving average to reduce volume jitter."""
    def __init__(self, alpha: float = 0.2):
        self.alpha  = alpha
        self._value = None

    def update(self, v: float) -> float:
        if self._value is None:
            self._value = v
        else:
            self._value = self.alpha * v + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None



def draw_volume_bar(frame, volume_pct: float, frame_w: int, frame_h: int):
    bx = frame_w - 40
    bt, bb = 100, frame_h - 100
    bh = bb - bt
    cv2.rectangle(frame, (bx, bt), (bx + 20, bb), (50, 50, 50), cv2.FILLED)
    fh  = int(bh * np.clip(volume_pct, 0, 100) / 100.0)
    col = (0, 255, 0) if volume_pct > 20 else (0, 80, 255)
    if fh > 0:
        cv2.rectangle(frame, (bx, bb - fh), (bx + 20, bb), col, cv2.FILLED)
    cv2.putText(frame, f"{int(volume_pct)}%", (bx - 10, bb + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "VOL", (bx - 5, bt - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_hand_skeleton(frame, landmarks, use_tasks, drawing_stuff,
                       frame_w: int, frame_h: int):
    """Draw the full skeleton if legacy API is active, else simple dots."""
    if not use_tasks and drawing_stuff is not None:
        mp_drawing, mp_draw_styles, HAND_CONNECTIONS = drawing_stuff
        from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
        proto = NormalizedLandmarkList()
        for lm in landmarks:
            nlm = proto.landmark.add()
            nlm.x, nlm.y, nlm.z = lm.x, lm.y, lm.z
        mp_drawing.draw_landmarks(
            frame, proto, HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style(),
        )
    else:
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)),
                       4, (0, 200, 255), cv2.FILLED)


# ──────────────────────────────────────────────
# 5. MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print("[Startup] Hand Gesture Volume Control")

    # Audio
    try:
        volume_iface, _ = setup_audio()
        print("[Audio] pycaw initialised successfully.")
    except RuntimeError as exc:
        print(exc); sys.exit(1)

    # Camera
    try:
        cap = setup_camera()
    except RuntimeError as exc:
        print(exc); sys.exit(1)

    # Read a real frame first so frame_w/frame_h always match the actual buffer.
    # cap.get() can lie on some Windows drivers (reports requested, not actual size).
    _ret, _probe = cap.read()
    if not _ret or _probe is None or _probe.size == 0:
        print("[Camera] Could not read probe frame – aborting.")
        cap.release(); sys.exit(1)
    frame_h, frame_w = _probe.shape[:2]
    print(f"[Camera] Actual resolution: {frame_w}x{frame_h}")

    # MediaPipe
    try:
        detector, use_tasks, drawing_stuff = setup_mediapipe()
    except RuntimeError as exc:
        print(exc); cap.release(); sys.exit(1)

    # State
    smoother       = Smoother(alpha=0.2)
    MIN_DIST       = 30    # pixels -> 0% volume
    MAX_DIST       = 250   # pixels -> 100% volume

    print("[Loop] Starting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        # Guard against malformed frames (step-mismatch / empty buffer)
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.05)
            continue
        if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
            # Driver occasionally returns a differently-sized frame; update dims
            frame_h, frame_w = frame.shape[:2]

        frame = cv2.flip(frame, 1)  # mirror view

        current_vol  = get_volume(volume_iface)
        current_mute = is_muted(volume_iface)

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detect_hands(detector, rgb, use_tasks)

        if landmarks is not None:
            # Draw skeleton / dots
            draw_hand_skeleton(frame, landmarks, use_tasks, drawing_stuff,
                               frame_w, frame_h)

            thumb = get_landmark_px(landmarks[4], frame_w, frame_h)
            index = get_landmark_px(landmarks[8], frame_w, frame_h)
            mid   = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)

            # Visual connectors
            cv2.circle(frame, thumb, 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, index, 10, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, thumb, index, (255, 0, 255), 3)
            cv2.circle(frame, mid,  8,  (0, 255, 0),   cv2.FILLED)

            # Map distance to volume
            dist      = euclidean_distance(thumb, index)
            vol_pct   = float(np.clip(
                np.interp(dist, [MIN_DIST, MAX_DIST], [0.0, 100.0]), 0.0, 100.0
            ))
            vol_smooth = smoother.update(vol_pct)

            if not current_mute:
                set_volume(volume_iface, vol_smooth)
                current_vol = vol_smooth

            cv2.putText(frame, f"dist: {int(dist)}px",
                        (mid[0] + 12, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


        else:
            smoother.reset()

        # UI
        display_vol = current_vol if not current_mute else 0.0
        draw_volume_bar(frame, display_vol, frame_w, frame_h)

        status = "MUTED" if current_mute else f"Vol: {int(current_vol)}%"
        cv2.putText(frame, status,
                    (frame_w // 2 - 60, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if current_mute else (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit",
                    (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Hand Gesture Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Loop] Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if hasattr(detector, 'close'):
        detector.close()
    print("[Shutdown] Done.")


if __name__ == "__main__":
    main()