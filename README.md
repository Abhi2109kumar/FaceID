# FaceID Liveness & Identity System (RGB Only)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![MediaPipe](https://img.shields.io/badge/Engine-MediaPipe-brightgreen.svg)

A professional-grade, software-based face recognition system with multi-modal liveness detection. This system is designed to provide secure biometric authentication using standard webcams, eliminating the need for expensive IR or depth-sensing hardware.

---

## 🏗️ Directory Structure

```text
FaceID/
├── data/                       # Local identity storage
│   └── users.json              # Encrypted face signatures
├── src/
│   ├── capture/
│   │   └── video_stream.py     # Threaded low-latency capture
│   ├── detection/
│   │   └── face_detector.py    # MediaPipe Face Mesh integration
│   ├── liveness/
│   │   ├── active/
│   │   │   └── challenge.py    # Randomized behavioral controller
│   │   └── passive/
│   │       └── texture_analysis.py # FFT & Laplacian frequency monitor
│   ├── recognition/
│   │   └── face_recognizer.py  # Geometric signature & pose-aware matching
│   └── utils/
│       ├── database.py         # JSON-based persistent storage
│       └── geometry.py         # SolvePnP & Head Pose estimation
├── main.py                     # Application entry point & State Machine
├── requirements.txt            # Dependency manifest
└── README.md                   # System documentation
```

---

## 🛠️ Technical Working Principle

The system employs a **Three-Layer Security Pipeline** to verify identity and ensure "proof of life."

### 1. Passive Liveness (Continuous Monitoring)
- **Texture Analysis:** Uses Laplacian Variance to calculate image sharpness, rejecting blurred low-res photos.
- **Frequency Analysis (FFT):** Performs a Fast Fourier Transform on the face ROI. It identifies high-frequency Moire patterns and "digital grids" typical of smartphone screens or tablets, which are invisible to the naked eye but distinct in the frequency domain.

### 2. Active Liveness (Behavioral Challenge)
- **Challenge-Response:** Randomly requests actions (Blinks, Head Turns).
- **Relative Pose Tracking:** Instead of checking for absolute angles, the system captures the user's initial orientation and measures the **angular delta**. This prevents static photos or "tilting images" from passing, as it requires true 3D volumetric rotation.

### 3. Identity matching (Geometric Signature)
- **Landmark Encoding:** Extracts 40+ high-precision landmarks.
- **Normalization:** Landmark coordinates are normalized relative to the nose tip and scaled by the inter-ocular distance (face width), making the signature invariant to distance from the camera.
- **Pose-Aware Buffering:** During challenges, the system buffers face signatures and automatically selects the most **front-facing** sample for comparison against the database.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A standard RGB Webcam

### Installation
1. **Clone & Navigate:**
   ```powershell
   cd FaceID
   ```
2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Operational Workflow
1. **Launch:** Run `python main.py`.
2. **Register ('r'):** Enter your name in the terminal. The system will run a liveness check. Once passed, your geometric signature is saved to `users.json`.
3. **Login ('l'):** The system will verify your liveness first. If successful, it will scan the database and authenticate your identity.

---

## 📊 Security Benchmarks

| Threat | Defense Layer | Mitigation Strategy |
| :--- | :--- | :--- |
| **Printed Photo** | Active Liveness | Fails EAR (Blink) and SolvePnP (3D Pose). |
| **Video Replay** | Passive Liveness | FFT detects Moire patterns and LCD refresh artifacts. |
| **3D Mask** | Passive Liveness | Frequency analysis detects unnatural skin texture & lack of pore-level detail. |
| **Static Image Tilt**| Relative Pose | Fails to produce the correct 3D geometric landmark shift. |

---

## 📄 License
This project is licensed under the MIT License. Developed for research in biometric security and agentic automation.