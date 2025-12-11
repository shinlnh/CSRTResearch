# 🚀 Updated CSRT Tracker (C++17 + PyTorch)

![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv&logoColor=white) ![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)

🎯 Dual-branch CSRT tracker that fuses classic HOG/ColorNames with deep VGG16 features. Includes a PyTorch training pipeline to learn the CorrProject projection and Adaptive Gating, then export to ONNX for the C++ runtime.

## ✨ Highlights
- 🔀 **Hybrid filters**: h_csrt (HOG/CN) blended with h_deep (projected VGG16) via adaptive α.
- 🧠 **Learned projection**: CorrProject (1×1 conv stack) maps 512→31 channels to match CSRT space.
- 🎚️ **Adaptive gating**: ONNX gating net estimates α from context (fallback to fixed α).
- 🧽 **Mask-aware DCF/ADMM**: spatial masks to suppress background leakage.
- 🎥 **Real-time C++**: OpenCV DNN backend; demo binary ready to run.
- 🧪 **Training suite**: PyTorch scripts for datasets, loss, solver, export to ONNX.

## 📂 Repo Map
```
update_csrt/
├─ inc/                # C++ headers (Config, trackers, extractors, solver, masks)
├─ src/                # C++ implementations + demo main.cpp
├─ models/             # Expected ONNX models (vgg16_conv4_3.onnx, corr_project.onnx, adaptive_gating.onnx)
├─ checkpoints/, runs/ # PyTorch training logs/checkpoints
├─ feature_extractor.py
├─ corr_project.py     # CorrProject, AdaptiveGating, HybridFilter (PyTorch)
├─ dcf_solver.py, segmentation.py, tracker.py, train.py, test.py
└─ CMakeLists.txt      # C++ build
```

## 🛠️ Build & Run (C++)
Prereqs: C++17 toolchain, OpenCV built with `opencv_contrib` (for CSRT), ONNX files in `update_csrt/models/`.
```powershell
cmake -S update_csrt -B build
cmake --build build --config Release
.\build\updated_csrt_demo.exe --camera 0 --display  # example flags; adjust to your pipeline
```
Key C++ config: `update_csrt/inc/Config.hpp` (HOG params, α limits, ADMM, mask options, ONNX paths). Print/validate helpers are included; ensure the `use_rescue` flag referenced in `print()` exists before enabling.

## 🧠 Train / Export (PyTorch)
```powershell
cd update_csrt
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Train CorrProject + gating on your data
python train.py
# Evaluate
python test.py --checkpoint checkpoints/...
# Export ONNX for C++
python ..\export_models_to_onnx.py
```
Main knobs: `update_csrt/config.py` (`PCSRTConfig`) mirrors the C++ config. Dataset root defaults to `otb100/OTB-dataset/OTB100` (set `sequences` to subset if needed).

## 🧭 Tracking Flow (C++)
1) Extract template patch → HOG/CN + VGG16 deep features.  
2) Apply mask, project deep features with CorrProjection ONNX.  
3) Solve DCF/ADMM for h_csrt and h_deep; blend with α (adaptive/fixed).  
4) For each frame: crop search region → dual responses → adaptive α → fused response peak → bbox update → filter refresh.

## 📌 Notes
- Models live under `update_csrt/models/`; adjust paths in `Config.hpp` if you relocate them.
- If adaptive gating model is missing, the tracker falls back to fixed `alpha_default`.
- Verbose logs and visualizations can be toggled in `Config.hpp`; Python side mirrors these in `PCSRTConfig`.

Enjoy hacking on the tracker! 🎉
