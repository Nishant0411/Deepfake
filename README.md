# DeepGuard — Multimodal Deepfake & Voice Clone Detection System

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Async_ASGI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Inference_Mode-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-ViT_%7C_Wav2Vec2-FFD21E?logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-YuNet_ONNX-5C3EE8?logo=opencv&logoColor=white)

**DeepGuard** is an asynchronous, low-latency forensic AI platform engineered to detect **manipulated facial videos**, **localized lip-sync reenactments**, and **AI-cloned voice calls** in under 3 seconds on local hardware.

Rather than relying on black-box classifiers alone, DeepGuard fuses deep transformer architectures (**Vision Transformer** and **Wav2Vec 2.0**) with deterministic frequency-domain and acoustic signal physics (**2D FFT**, **Laplacian edge variance**, **vocal fold micro-jitter**, and **Wiener spectral flatness**) to eliminate single-modality blind spots and provide transparent forensic telemetry.

---

## Core Features

* **Dual-Region Spatial RoI Extraction:** Evaluates the full face (with a 30% boundary margin) alongside an isolated lower-face mouth crop (`52%–100%` vertical height) in parallel to catch subtle talking-head and lip-sync manipulations (e.g., Wav2Lip, SadTalker) where the upper face remains authentic.
* **Hybrid Visual Forensics:** Pairs `ViT-Base/16` global multi-head self-attention across 196 patches with radial **2D Fast Fourier Transform (FFT)** high-frequency energy ratios and **Laplacian gradient variance** to expose generative upsampling checkerboards and blending seams.
* **Top-K Multiple Instance Learning (MIL) Pooling:** Aggregates frame-level predictions by evaluating the top 30% most suspicious keyframes (`K = 0.30 * N`), suppressing false positives from camera motion blur while reliably flagging short-duration manipulations.
* **Unified Tensor Batch Parallelism:** Stacks all 24 full-face crops and 24 mouth crops into a single contiguous tensor batch under `torch.inference_mode()`, cutting video inference time from 20+ seconds down to **1.8–2.8 seconds**.
* **Universal Audio & Voice Call Forensics:** Transcodes `.ogg` (WhatsApp), `.m4a` (iOS), `.mp3`, and `.wav` files into 16 kHz mono waveforms via an in-memory **FFmpeg** pipeline, filtering silent pauses with **Voice Activity Gating (VAG)** before scoring neural vocoder artifacts and biological pitch jitter.
* **Interactive Forensic Dashboard:** Lightweight web UI (`static/demo.html`) displaying real-time authenticity verdicts, confidence scores, frame-level telemetry, and acoustic naturalness indicators.

---

## System Architecture

```text
                       ┌───────────────────────────┐
                       │    Media Upload (UI/API)  │
                       └─────────────┬─────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
      [ Visual Pipeline ]                     [ Audio Pipeline ]
  24 Uniform Keyframes (OpenCV)          FFmpeg 16 kHz Mono Transcoding
                 │                                       │
      OpenCV YuNet (ONNX FPN)               Voice Activity Gating (VAG)
    (Face + 30% Margin & Mouth RoI)                      │
                 │                           ┌───────────┴───────────┐
      ┌──────────┴──────────┐                ▼                       ▼
      ▼                     ▼          Wav2Vec 2.0             Signal Forensics
ViT-Base/16          2D FFT & Laplacian (Transformer)       (Autocorrelation Jitter
(Unified Batch)      (Frequency Checks)                         & Wiener Entropy)
      │                     │                │                       │
      └──────────┬──────────┘                └───────────┬───────────┘
                 ▼                                       ▼
      Top-K MIL Anomaly Pooling               Acoustic Score Fusion
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                     Forensic Verdict & Telemetry JSON
```

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & API** | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| **Visual Deep Learning** | PyTorch, Hugging Face Transformers (`ViT-Base/16`), OpenCV (`YuNet` ONNX) |
| **Audio Deep Learning** | Wav2Vec 2.0 (`mo-thecreator/Deepfake-audio-detection`), `imageio-ffmpeg` |
| **Signal Forensics** | NumPy (2D FFT, STFT Wiener Entropy, Short-Time Autocorrelation), OpenCV Laplacian |
| **Frontend Dashboard** | HTML5, CSS3, Vanilla JavaScript (Async Fetch API) |

---

## Project Structure

```text
Deepfake/
├── main.py                      # FastAPI server, API routing, and in-memory FFmpeg audio transcoding
├── analyze_frames.py            # Standalone frame-level diagnostic and inspection script
├── requirements.txt             # Pinned Python dependencies
├── .gitignore                   # Excludes virtual environments, local model weights, and temp media
├── services/
│   ├── video_processor.py       # Uniform keyframe seeking, YuNet face detection, and dual-region cropping
│   ├── detector.py              # ViT batch inference, 2D FFT/Laplacian metrics, and Top-K MIL pooling
│   └── audio_detector.py        # Wav2Vec 2.0 inference, Voice Activity Gating, and micro-jitter scoring
└── static/
    └── demo.html                # Dual-pipeline forensic telemetry web interface
```

---

## Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/Nishant0411/Deepfake.git](https://github.com/Nishant0411/Deepfake.git)
cd Deepfake
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** On the first run, the OpenCV YuNet ONNX detector and Hugging Face model weights are downloaded and cached locally on disk, enabling full offline execution on subsequent runs.

### 4. Start the FastAPI Server
**Windows (PowerShell):**
```powershell
.\env\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

**macOS / Linux:**
```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

### 5. Launch the Web Interface
Open your browser and navigate to:
```text
[http://127.0.0.1:8001/demo](http://127.0.0.1:8001/demo)
```

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/demo` | Serves the interactive multimodal forensic dashboard. |
| `POST` | `/api/v1/detect` | Accepts video or image files (`.mp4`, `.mov`, `.avi`, `.jpg`, `.png`) and returns aggregated detection telemetry. |
| `POST` | `/api/v1/detect-audio` | Accepts audio files (`.wav`, `.mp3`, `.m4a`, `.ogg`) and returns synthetic voice probabilities and pitch jitter metrics. |

### Sample Visual Telemetry Output
```json
{
  "verdict": "FAKE",
  "confidence": 89.4,
  "multimodal_telemetry": {
    "visual_vit_prob": 91.2,
    "frequency_fft_anomaly": 78.6,
    "boundary_consistency": 87.5
  },
  "model_info": {
    "architecture": "Vision Transformer (ViT-Base/16) + Top-K Dual-Region Forensics",
    "device": "cuda"
  }
}
```
