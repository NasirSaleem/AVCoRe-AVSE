# AVCoRe: Audio-Visual Codec Resynthesis & Waveform Fusion

Official PyTorch and ONNX implementation for **AVCoRe**: *Entropy-Informed Fusion of Waveform and Neural Codec Reconstructions for Audio-Visual Speech Enhancement*.

📌 Overview

AVCoRe is a dual-path Audio-Visual Speech Enhancement (AVSE) framework that combines an **observation-anchored continuous waveform estimate** with a **discrete neural-codec reconstruction** (EnCodec). The two pathways are combined using a learned segment-level fusion gate driven by codec features and token-prediction entropy statistics ($\mu_H, \sigma_H$).

Key Features
Dual-Path Architecture:** Mitigates acoustic residual noise (via continuous masking) and avoids neural codec hallucination (via discrete token entropy gating).
Uncertainty-Informed Fusion Gate:** Predicts sample-dependent fusion weight $\alpha \in [0, 1]$ based on token entropy bounds.
On-Device Real-Time Performance:** Exportable to FP32 ONNX Runtime, achieving $\text{RTF} < 1.0$ on mobile processors (Samsung Galaxy S22 Ultra and Galaxy A17).

📊 Dataset Access

Model evaluation and training use the COG-MHEAR Challenge dataset (derived from LRS3):
Official Benchmark Dataset: Download the COG-MHEAR evaluation audio-visual data directly from the [Official COG-MHEAR GitHub Repository](https://github.com/COG-MHEAR).

📱 Mobile Deployment (Android Studio)

Due to GitHub repository file size restrictions, the complete, ready-to-build **Android Studio Project** (including native C++ ONNX Runtime bindings and APK setup) is hosted externally.

📁 **Download Link:** Please refer to [`android_download_link.txt`](./android_download_link.txt) in this repository to access the download link and mobile build instructions.

📂 Repository Structure

├── models/
│   ├── avcore_model.py        # Complete dual-path AVCoRe architecture
│   ├── fusion_gate.py         # Uncertainty-informed entropy fusion gate
│   └── encoders.py            # Audio and visual feature extraction backbones
├── onnx/
│   ├── export_onnx.py         # Script to export PyTorch model to FP32 ONNX
│   └── run_onnx_benchmark.py  # Latency & RTF benchmarking script
├── evaluation/
│   └── metrics.py             # Objective metrics (PESQ, STOI, SI-SDR)
├── android_download_link.txt  # Link & instructions for the Android Studio project
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
