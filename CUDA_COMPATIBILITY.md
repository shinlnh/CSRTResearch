# 🎯 CUDA Compatibility cho RTX 5060

## GPU Information
- **Model**: NVIDIA GeForce RTX 5060
- **Architecture**: Ada Lovelace (hoặc Blackwell nếu là 50xx series mới)
- **Compute Capability**: 8.9
- **Driver**: 591.59

## CUDA Versions Tương Thích

### ✅ Driver 591.59 Support:
- **CUDA 13.1** (Recommended - Hiển thị trong nvidia-smi)
- **CUDA 13.0**
- **CUDA 12.x** (12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6)
- **CUDA 11.8** (Minimum for Ada Lovelace)

### 🎯 RECOMMENDED cho anh:

**CUDA 12.6** hoặc **CUDA 13.1**

**Tại sao?**
1. RTX 5060 là GPU mới → cần CUDA version mới
2. Driver 591.59 là driver mới nhất → support CUDA 13.1
3. CUDA 12.6 stable hơn CUDA 13.x cho production

---

## Download Links

### Option 1: CUDA 12.6 (STABLE - Khuyên dùng)
```
https://developer.nvidia.com/cuda-12-6-0-download-archive
```

### Option 2: CUDA 13.1 (LATEST - Matching driver)
```
https://developer.nvidia.com/cuda-downloads
```

---

## Quick Install Command

```powershell
# Download CUDA 12.6
$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe"
$installer = "E:\cuda_12.6.0_installer.exe"

Invoke-WebRequest -Uri $cudaUrl -OutFile $installer
Start-Process $installer
```

---

## Compute Capability Check

RTX 5060 → **Compute Capability 8.9**

Khi build OpenCV với vcpkg, dùng:
```cmake
-DCUDA_ARCH_BIN="8.9"
```

---

## Summary

| CUDA Version | Compatibility | Stability | Recommendation |
|--------------|---------------|-----------|----------------|
| CUDA 13.1 | ✅ Full | ⚠️ Beta | For bleeding edge |
| CUDA 12.6 | ✅ Full | ✅ Stable | **👍 BEST** |
| CUDA 12.1 | ✅ Good | ✅ Stable | Good alternative |
| CUDA 11.8 | ⚠️ Minimum | ✅ Very Stable | Too old |

---

## 🎯 Em đề xuất:

**Cài CUDA 12.6** thay vì 12.1 (đã download)
- Stable hơn
- Full support RTX 5060
- Tương thích tốt với OpenCV

Anh muốn cài CUDA 12.6 không?
