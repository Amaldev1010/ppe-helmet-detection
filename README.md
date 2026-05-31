PPE Helmet Detection — Edge Inference with YOLOv8n + ONNX FP16

Real-time PPE (Personal Protective Equipment) helmet detection system optimized for edge deployment using YOLOv8n trained on 14,748 images and converted to ONNX FP16 format.

---

Problem Statement

In industrial and construction environments, ensuring workers wear helmets is critical for safety compliance. This project builds a real-time edge-deployable detector that identifies:

- `helmet` — worker is wearing a helmet ✅
- `head` — worker is NOT wearing a helmet ⚠️

---

Dataset

- **Source:** [Hard Hat Workers — Roboflow Universe](https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/12)
- **Version:** 12
- **License:** Public Domain
- **Total Images:** 16,867 (Train: 14,748 | Val: 1,413 | Test: 706)
- **Classes:** 2 (`head`, `helmet`)

---

Project Structure

```
ppe-helmet-detection/
├── README.md
├── live_inference.py          # Edge inference script with real-time metrics
├── training_notebook.ipynb    # Full training notebook (Kaggle)
```

---

Phase 1: Training (FP32 Baseline)

- **Model:** YOLOv8n (nano) — lightest YOLOv8 variant
- **Framework:** Ultralytics 8.4.57
- **Hardware:** Kaggle T4 GPU
- **Epochs:** 32 (early stopping at epoch 22, best saved)
- **Image size:** 640×640
- **Batch size:** 32

Baseline Results

| Class | mAP50 | mAP50-95 |
|---|---|---|
| All | 0.965 | 0.649 |
| head | 0.950 | 0.642 |
| helmet | 0.979 | 0.655 |

---

Phase 2: Edge Conversion & Quantization

Format: ONNX
Quantization: FP16

**Why FP16 over INT8?**

FP16 (16-bit floating point) was chosen over INT8 (8-bit integer) for the following reasons:

1. **Minimal accuracy loss** — FP16 preserves the numerical range of the original FP32 weights, resulting in less than 2% mAP drop compared to 3–5% typical for INT8.
2. **No calibration data required** — INT8 quantization requires a representative calibration dataset to determine optimal quantization ranges. FP16 does not.
3. **Safe for small models** — YOLOv8n is already a compact model (6.2MB). INT8 quantization on very small models risks significant accuracy degradation due to reduced numerical precision in already-compressed weights.
4. **Universal compatibility** — ONNX FP16 runs on CPU, GPU, and most edge hardware without additional drivers.

---

Performance Benchmark Table

| Metric | FP32 `.pt` (Baseline) | FP16 `.onnx` (Edge) | Change |
|---|---|---|---|
| **Model Size (MB)** | 6.25 MB | 6.17 MB | -1.3% |
| **mAP50-95** | 0.6490 | 0.6318 | -1.72% drop |
| **mAP50** | 0.9650 | 0.9638 | -0.12% drop |
| **Helmet mAP50** | 0.979 | 0.977 | -0.2% drop |
| **Head mAP50** | 0.950 | 0.951 | +0.1% |
| **Inference FPS (local CPU)** | ~3.1ms/img (GPU val) | **11.3 FPS avg** | Edge optimized |
| **Pre-process latency** | — | ~11.6 ms | — |
| **Inference latency** | — | ~83.2 ms | — |
| **NMS post-process** | — | ~44.9 ms | — |
| **Quantization** | FP32 | FP16 | 50% bit reduction |
| **Format** | PyTorch `.pt` | ONNX | Edge compatible |

> Note: Baseline inference measured on T4 GPU during validation. Edge FPS measured on local CPU machine (Intel, no GPU) running `live_inference.py` with webcam input.

---

Phase 3: Inference Script

`live_inference.py` loads the ONNX FP16 model and runs real-time inference on a webcam feed, overlaying:

- Bounding boxes with class labels and confidence scores
- Live FPS (excluding rendering time)
- Pre-processing latency (ms)
- Inference latency (ms)
- Post-processing / NMS latency (ms)

Running locally

```bash
pip install onnxruntime opencv-python numpy
python live_inference.py
```

Press `Q` to quit. Average FPS printed in terminal on exit.

---

Model Weights

| File | Format | Size | Link |
|---|---|---|---|
| `best.pt` | PyTorch FP32 | 6.25 MB | [Google Drive] https://drive.google.com/drive/folders/1HDL3n6MczEmHk_P9vXkfTsVfB6XYkIFB?usp=sharing |
| `best.onnx` | ONNX FP16 | 6.17 MB | [Google Drive] https://drive.google.com/drive/folders/1HDL3n6MczEmHk_P9vXkfTsVfB6XYkIFB?usp=sharing |



---

Trade-off Analysis

| Factor | FP32 Baseline | FP16 ONNX Edge |
|---|---|---|
| Accuracy (mAP50) | 96.5% | 96.4% |
| Model size | 6.25 MB | 6.17 MB |
| Deployment | GPU server | CPU edge device |
| RAM usage | Higher | Lower |
| Compatibility | PyTorch only | Universal (ONNX) |
| Quantization loss | None | 1.72% mAP50-95 |

**Conclusion:** The FP16 ONNX model achieves near-identical accuracy (only 0.12% mAP50 drop) while being universally deployable on edge devices without requiring a GPU. The trade-off is minimal accuracy loss for maximum portability and reduced memory footprint.

---

Requirements

```
ultralytics>=8.4.0
onnxruntime>=1.26.0
opencv-python>=4.8.0
numpy>=1.24.0
```
