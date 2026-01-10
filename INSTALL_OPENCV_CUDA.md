# 🚀 CÁCH DỄ NHẤT: Cài OpenCV CUDA bằng vcpkg

## Bước 1: Install vcpkg (Package manager cho C++)

```powershell
# Clone vcpkg
cd E:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Add to PATH (optional)
$env:PATH += ";E:\vcpkg"
```

## Bước 2: Install OpenCV với CUDA

```powershell
# Install OpenCV with CUDA support
.\vcpkg install opencv[cuda]:x64-windows

# Hoặc với nhiều features:
.\vcpkg install opencv[core,cuda,cudnn,dnn,ffmpeg,contrib]:x64-windows
```

⏰ **Lưu ý**: Quá trình này mất **2-4 giờ** vì vcpkg sẽ build từ source!

## Bước 3: Integrate với CMake

```powershell
# Integrate vcpkg với Visual Studio/CMake
.\vcpkg integrate install
```

## Bước 4: Update CMakeLists.txt

```cmake
# Trong update_csrt/CMakeLists.txt
set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")
find_package(OpenCV REQUIRED)
```

---

# ⚡ GIẢI PHÁP NHANH HƠN: Docker với OpenCV CUDA

Nếu anh không muốn build, dùng Docker image có sẵn:

```powershell
# Pull image với OpenCV CUDA
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Hoặc image có sẵn OpenCV
docker pull opencvcourses/opencv-docker
```

---

# 🎯 RECOMMENDATION CHO ANH

**Anh đang dùng MinGW** → Có 3 lựa chọn:

### Option A: Giữ MinGW + Multi-Threading (⚡ NHANH & DỄ - Đã có!)
- ✅ 20 threads parallel
- ✅ Không cần GPU
- ✅ Speedup ~20x
- ⏱️ **Đang chạy rồi!**

### Option B: Chuyển sang MSVC + CUDA (⚡⚡⚡ NHANH NHẤT)
- Cài Visual Studio 2022 Community (free)
- Dùng vcpkg install OpenCV CUDA
- Rebuild project với MSVC
- ⏰ Mất 1 ngày setup

### Option C: MinGW + OpenCL (UMat) (⚡⚡ VỪA PHẢI)
- OpenCV MinGW có thể có OpenCL
- Không cần rebuild OpenCV
- Chỉ sửa code dùng UMat
- ⏰ 30 phút implement

---

# 🔍 CHECK: Anh có GPU NVIDIA không?

```powershell
nvidia-smi
```

Nếu không có output → Không có NVIDIA GPU → **CUDA vô dụng!**

---

# ✨ Em đề xuất cho anh:

**Bước 1**: Để multi-threading chạy xong (đang chạy)
**Bước 2**: Check kết quả xem speedup có đủ không
**Bước 3**: Nếu vẫn chậm, em implement OpenCL (UMat) - dễ hơn CUDA nhiều!

Anh muốn tiếp tục theo hướng nào?
1. Đợi kết quả multi-threading
2. Install Visual Studio + vcpkg để dùng CUDA
3. Try OpenCL (UMat) ngay - không cần GPU driver đặc biệt
