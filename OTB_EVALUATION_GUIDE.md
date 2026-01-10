# OTB Evaluation Guide - Fast Comparison Workflow

## Workflow Overview

Để so sánh `update_csrt` với `pure_csrt` một cách **nhanh nhất**, làm theo 2 bước:

### Bước 1: Chạy Pure CSRT (1 lần duy nhất)
```bash
cd pure_csrt_detail/build
./otb_compare --dataset-root ../../otb100
```

Kết quả sẽ lưu vào: `pure_csrt_detail/auc_compare.csv`

⏱️ **Thời gian**: ~30-60 phút (tùy dataset)

### Bước 2: Chạy Update CSRT (nhiều lần)
```bash
cd update_csrt/build
./otb_compare --dataset-root ../../otb100
```

Kết quả sẽ lưu vào: `update_csrt/auc_compare.csv`

⏱️ **Thời gian**: ~15-30 phút (chỉ chạy 1 tracker)

**Lợi ích**: Mỗi lần thay đổi `update_csrt`, chỉ cần chạy Bước 2 → **nhanh gấp đôi!**

---

## Advanced Options

### Update CSRT Options

```bash
# Sử dụng baseline từ file khác
./otb_compare --dataset-root ../../otb100 --pure-csv /path/to/baseline.csv

# Chạy cả 2 tracker (chậm hơn, để debug)
./otb_compare --dataset-root ../../otb100 --no-baseline

# Giới hạn frames để test nhanh
./otb_compare --dataset-root ../../otb100 --max-frames 100

# Đổi output file
./otb_compare --dataset-root ../../otb100 --output my_results.csv
```

### Pure CSRT Options

```bash
# Giới hạn frames
./otb_compare --dataset-root ../../otb100 --max-frames 100

# Đổi output file
./otb_compare --dataset-root ../../otb100 --output baseline.csv
```

---

## Output Format

Cả 2 file CSV có format giống nhau:

```csv
sequence,frames,auc_update,auc_pure,success50_update,success50_pure,precision20_update,precision20_pure,fps_update,fps_pure,delta_auc
Basketball,725,0.5234,0.5123,0.6543,0.6421,0.8765,0.8654,45.23,42.11,0.0111
OVERALL,..,...,...,...,...,...,...,...,...,...
```

**Metrics**:
- `auc`: Area Under Curve (success plot)
- `success50`: Success rate @ IoU=0.5
- `precision20`: Precision @ 20 pixels
- `fps`: Frames per second
- `delta_auc`: Improvement (positive = better)

---

## Workflow Comparison

### Traditional Way (Slow) ❌
```bash
# Mỗi lần test phải chạy cả 2 tracker
cd update_csrt/build
./otb_compare --dataset-root ../../otb100 --no-baseline  # ~60 phút
```

### Optimized Way (Fast) ✅
```bash
# Lần đầu: chạy pure_csrt
cd pure_csrt_detail/build
./otb_compare --dataset-root ../../otb100  # ~30 phút (1 lần duy nhất)

# Các lần sau: chỉ chạy update_csrt
cd ../../update_csrt/build
./otb_compare --dataset-root ../../otb100  # ~15 phút (nhiều lần)
```

**Tiết kiệm**: 50% thời gian mỗi lần test!

---

## Troubleshooting

### Warning: Baseline CSV not found
```
Warning: Baseline CSV not found: pure_csrt_detail/auc_compare.csv
Run pure_csrt_detail first to generate baseline.
Falling back to running both trackers (slower)...
```

**Giải pháp**: Chạy Bước 1 trước (pure_csrt_detail)

### Warning: Baseline incomplete
```
Warning: Baseline CSV missing sequence: Basketball
Warning: Baseline incomplete, running both trackers...
```

**Nguyên nhân**: File baseline bị lỗi hoặc thiếu sequences

**Giải pháp**: Chạy lại pure_csrt_detail với cùng dataset và --max-frames (nếu có)

---

## Tips

1. **Test nhanh**: Dùng `--max-frames 50` để test code trước khi chạy full dataset
2. **Parallel processing**: Code tự động dùng multi-threading, không cần config
3. **CUDA**: Nếu có GPU, tracker sẽ tự động dùng (check console output)
4. **Backup baseline**: Sao lưu `pure_csrt_detail/auc_compare.csv` trước khi chạy lại

---

## Example Session

```bash
# Lần đầu setup
cd pure_csrt_detail/build
cmake --build . --config Release
./otb_compare --dataset-root ../../otb100

# Test thay đổi trong update_csrt
cd ../../update_csrt
# ... edit code ...
cd build
cmake --build . --config Release
./otb_compare --dataset-root ../../otb100  # Fast! Chỉ ~15 phút

# Test nhanh với 100 frames
./otb_compare --dataset-root ../../otb100 --max-frames 100  # ~2-3 phút
```

**Enjoy fast iteration!** 🚀
