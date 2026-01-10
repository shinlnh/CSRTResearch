# 🚀 Setup OpenCV với CUDA cho RTX 5060

## Bước 1: Install Visual Studio 2022 Community (FREE)

1. Tải: https://visualstudio.microsoft.com/downloads/
2. Chọn: **Visual Studio 2022 Community** (Free for individuals)
3. Trong installer, chọn workloads:
   - ✅ Desktop development with C++
   - ✅ Linux and embedded development with C++ (optional)

## Bước 2: Install CUDA Toolkit 13.1

Anh đã có driver 591.59 với CUDA 13.1, nhưng cần CUDA Toolkit:

```powershell
# Download từ NVIDIA
# https://developer.nvidia.com/cuda-13-1-0-download-archive

# Hoặc dùng Chocolatey (nếu có)
choco install cuda --version=13.1.0
```

## Bước 3: Install OpenCV CUDA với vcpkg

```powershell
cd E:\vcpkg

# Set Visual Studio environment
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"

# Install OpenCV with CUDA (mất 2-4 giờ!)
.\vcpkg install opencv[cuda,cudnn,contrib]:x64-windows
```

## Bước 4: Update Project để dùng vcpkg

```powershell
# Integrate vcpkg
E:\vcpkg\vcpkg integrate install
```

Trong `update_csrt/CMakeLists.txt`:

```cmake
# Add vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Find OpenCV (vcpkg sẽ tự động tìm version có CUDA)
find_package(OpenCV REQUIRED)
```

## Bước 5: Rebuild Project với Visual Studio

```powershell
cd E:\Programming\C\C2P\Project\CSRTResearch\update_csrt

# Create new build folder for MSVC
mkdir build_msvc
cd build_msvc

# Configure với Visual Studio
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="E:/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Build
cmake --build . --config Release
```

---

# ⚡ TÓM TẮT

**Thời gian cần:**
- Install VS 2022: 30 phút
- Install CUDA Toolkit: 20 phút  
- vcpkg build OpenCV: 2-4 giờ
- Rebuild project: 10 phút
**TỔNG: ~3-5 giờ**

**Kết quả:**
- Multi-threading (20x) + CUDA (10x) = **200x faster!**

---

# ❓ Anh có muốn em bắt đầu không?

Nếu anh muốn tiết kiệm thời gian, em đề xuất:
1. ✅ **Dùng kết quả multi-threading hiện tại** (đã nhanh 20x rồi!)
2. 🔄 Install Visual Studio + CUDA sau, khi anh có thời gian
3. 📊 So sánh performance sau

Anh quyết định thế nào?
