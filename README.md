# Hand Gesture Volume Control for Windows


A Python-based Computer Vision application that allows users to control the system master volume using hand gestures via a webcam. By tracking the distance between the thumb and the index finger, it adjusts the audio levels in real-time.


## Features


- **Dual MediaPipe API Support:** Features a fallback chain that tries the new MediaPipe Tasks API (>= 0.10) first, and falls back to the legacy solutions API if unavailable.


- **Jitter Reduction:** Implements an Exponential Moving Average (Smoother class) to eliminate volume fluctuations caused by minor hand shaking.


- **Robust Camera Setup:** Scans multiple camera indices and backends (like `CAP_DSHOW`) to guarantee standard driver compatibility on Windows.


- **Windows Audio Integration:** Uses `pycaw` with a 3-method fallback chain to securely hook into the Windows Core Audio API.


## Requirements


You need to install the following dependencies:


```bash
pip install opencv-python mediapipe pycaw numpy comtypes

```markdown
## How it Works


The script processes frames from your webcam and looks for a hand skeleton.


### Expected Gesture


- **Volume Up/Down:** Move your Thumb (Landmark 4) and Index Finger (Landmark 8) closer or further apart.


- **Visual Feedback:** A dynamic volume bar and status text are drawn directly onto the mirrored camera preview screen.


### Logic Breakdown


- **Distance Mapping:** The Euclidean distance between the two fingertips is computed in pixels and mapped to a percentage scale (0% to 100%) using `numpy.interp`.


- **Dynamic Dimensions:** Automatically handles sudden camera frame-size switches to avoid step-mismatch assertion crashes.

## How to Run


### Steps


1. Save the code in a file named `volume_control.py`.


2. Open your terminal or command prompt with administrative privileges (recommended for Windows audio control).


3. Run the script:


   ```bash
   python volume_control.py
