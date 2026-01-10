# Build Instructions with vcpkg and CUDA

## Prerequisites

1. **vcpkg** installed and `VCPKG_ROOT` environment variable set
2. **OpenCV with CUDA** built and installed at `C:/opencv`
3. **Visual Studio 2019/2022** with C++ development tools
4. **Ninja build system** (install via vcpkg: `vcpkg install ninja`)
5. **CUDA Toolkit** and **cuDNN** installed

## Quick Build

### Using PowerShell Script (Recommended)

```powershell
# Build Release version with 8 parallel jobs
.\build_project.ps1 -BuildType Release -Jobs 8

# Clean build
.\build_project.ps1 -Clean -BuildType Release

# Build Debug version
.\build_project.ps1 -BuildType Debug
```

### Using CMake Presets

```powershell
# Configure
cmake --preset windows-release

# Build
cmake --build --preset windows-release

# Or in one step
cmake --preset windows-release && cmake --build --preset windows-release
```

## Manual Build Steps

### 1. Install Dependencies via vcpkg

```powershell
# Navigate to vcpkg directory
cd $env:VCPKG_ROOT

# Install Intel TBB for parallel processing
vcpkg install tbb:x64-windows

# Integrate vcpkg with system
vcpkg integrate install
```

### 2. Build Main Project

```powershell
# Configure
cmake --preset windows-release

# Build
cmake --build --preset windows-release --parallel 8
```

### 3. Build Subprojects

#### Pure CSRT Detail

```powershell
cd pure_csrt_detail
mkdir build -Force
cd build

cmake .. -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DOpenCV_DIR="C:/opencv/build" `
  -DUSE_CUDA=ON

cmake --build . --config Release --parallel 8
cd ../..
```

#### Update CSRT

```powershell
cd update_csrt
mkdir build -Force
cd build

cmake .. -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DOpenCV_DIR="C:/opencv/build" `
  -DUSE_CUDA=ON

cmake --build . --config Release --parallel 8
cd ../..
```

## Configuration Options

### CMake Variables

- `OpenCV_DIR`: Path to OpenCV build directory (default: `C:/opencv/build`)
- `USE_CUDA`: Enable CUDA support (default: `ON`)
- `CMAKE_BUILD_TYPE`: Build type - `Release` or `Debug`

### Enable/Disable Features

```powershell
# Build without CUDA
cmake --preset windows-release -DUSE_CUDA=OFF

# Use different OpenCV location
cmake --preset windows-release -DOpenCV_DIR="D:/custom/opencv/build"
```

## Multi-Threading Support

The project is configured with multiple threading backends:

1. **OpenMP** - Compiler-level parallelization
   - MSVC: `/openmp` flag enabled
   - GCC/Clang: `-fopenmp` flag enabled

2. **Intel TBB** - High-performance parallel algorithms
   - Automatically linked if available via vcpkg
   - Provides efficient task scheduling

3. **std::thread** - Standard C++ threading
   - Always available via CMake `Threads::Threads`

## CUDA Support

When OpenCV is built with CUDA:

- `HAVE_OPENCV_CUDA` definition is set
- CUDA include directories are added
- GPU acceleration is available for:
  - Image processing operations
  - Template matching
  - Feature extraction

## Verify Build

### Check CUDA Support

```powershell
# Run verification
.\build\windows-release\csrtpure.exe

# Check OpenCV build info
cd C:\opencv\build\bin
.\opencv_version.exe -v
```

### Test Executables

```powershell
# Main executable
.\build\windows-release\csrtpure.exe

# Pure CSRT demos
.\pure_csrt_detail\build\csrt_demo.exe
.\pure_csrt_detail\build\otb_eval.exe
.\pure_csrt_detail\build\otb_compare.exe

# Update CSRT demos
.\update_csrt\build\csrt_demo.exe
.\update_csrt\build\otb_eval.exe
.\update_csrt\build\otb_compare.exe
```

## Troubleshooting

### OpenCV Not Found

```
CMake Error: Could not find OpenCV
```

**Solution**: Ensure OpenCV_DIR points to the correct location:
```powershell
cmake --preset windows-release -DOpenCV_DIR="C:/opencv/build"
```

### CUDA Not Detected

```
OpenCV without CUDA support
```

**Solution**: Check OpenCV was built with CUDA:
```powershell
cd C:\opencv\build\bin
.\opencv_version.exe -v | Select-String -Pattern "CUDA"
```

### vcpkg Not Found

```
CMake Error: CMAKE_TOOLCHAIN_FILE not found
```

**Solution**: Set VCPKG_ROOT environment variable:
```powershell
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
# Or set permanently in System Properties
```

### TBB Linking Errors

**Solution**: Install TBB via vcpkg:
```powershell
vcpkg install tbb:x64-windows
vcpkg integrate install
```

## Performance Tips

1. **Use Release build** for maximum performance
2. **Enable all CPU cores**: Build with `-j` flag
3. **CUDA acceleration**: Ensure GPU is not in use by other applications
4. **TBB parallelism**: Optimal for OTB benchmark evaluation with multiple videos

## Output Structure

```
build/
  windows-release/
    csrtpure.exe
pure_csrt_detail/
  build/
    csrt_demo.exe
    otb_eval.exe
    otb_compare.exe
    test_basketball.exe
update_csrt/
  build/
    csrt_demo.exe
    otb_eval.exe
    otb_compare.exe
```
