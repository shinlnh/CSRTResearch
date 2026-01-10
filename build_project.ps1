# Build script for CSRT Research Project with vcpkg and CUDA
param(
    [string]$BuildType = "Release",
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [switch]$Clean = $false,
    [int]$Jobs = 8
)

Write-Host "=== CSRT Research Build Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if vcpkg is available
if (-not $VcpkgRoot -or -not (Test-Path $VcpkgRoot)) {
    Write-Host "Error: VCPKG_ROOT not set or vcpkg not found!" -ForegroundColor Red
    Write-Host "Please set VCPKG_ROOT environment variable or provide -VcpkgRoot parameter" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using vcpkg from: $VcpkgRoot" -ForegroundColor Green

# Set environment variables
$env:VCPKG_ROOT = $VcpkgRoot

# Check if OpenCV CUDA build exists
$OpenCVDir = "C:/opencv/build"
if (-not (Test-Path $OpenCVDir)) {
    Write-Host "Warning: OpenCV CUDA build not found at $OpenCVDir" -ForegroundColor Yellow
    Write-Host "Make sure you have built OpenCV with CUDA support first" -ForegroundColor Yellow
} else {
    Write-Host "Found OpenCV CUDA build at: $OpenCVDir" -ForegroundColor Green
}

# Clean build directories if requested
if ($Clean) {
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    if (Test-Path "pure_csrt_detail/build") {
        Remove-Item -Recurse -Force "pure_csrt_detail/build"
    }
    if (Test-Path "update_csrt/build") {
        Remove-Item -Recurse -Force "update_csrt/build"
    }
}

# Configure build preset based on build type
$PresetName = "windows-$($BuildType.ToLower())"

Write-Host ""
Write-Host "=== Building Main Project ===" -ForegroundColor Cyan
Write-Host "Preset: $PresetName" -ForegroundColor White

# Configure with CMake
Write-Host "Configuring CMake..." -ForegroundColor Yellow
cmake --preset $PresetName
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "Building project..." -ForegroundColor Yellow
cmake --build --preset $PresetName --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Build pure_csrt_detail subproject
Write-Host ""
Write-Host "=== Building Pure CSRT Detail ===" -ForegroundColor Cyan
Push-Location pure_csrt_detail

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location build

Write-Host "Configuring Pure CSRT Detail..." -ForegroundColor Yellow
cmake .. -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DCMAKE_TOOLCHAIN_FILE="$VcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
    -DOpenCV_DIR="C:/opencv/build" `
    -DUSE_CUDA=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow
cmake --build . --config $BuildType --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Build update_csrt subproject
Write-Host ""
Write-Host "=== Building Update CSRT ===" -ForegroundColor Cyan
Push-Location update_csrt

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location build

Write-Host "Configuring Update CSRT..." -ForegroundColor Yellow
cmake .. -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DCMAKE_TOOLCHAIN_FILE="$VcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
    -DOpenCV_DIR="C:/opencv/build" `
    -DUSE_CUDA=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow
cmake --build . --config $BuildType --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

Write-Host ""
Write-Host "=== Build Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Executables built:" -ForegroundColor Cyan
Write-Host "  - build/$PresetName/csrtpure.exe" -ForegroundColor White
Write-Host "  - pure_csrt_detail/build/csrt_demo.exe" -ForegroundColor White
Write-Host "  - pure_csrt_detail/build/otb_eval.exe" -ForegroundColor White
Write-Host "  - pure_csrt_detail/build/otb_compare.exe" -ForegroundColor White
Write-Host "  - update_csrt/build/csrt_demo.exe" -ForegroundColor White
Write-Host "  - update_csrt/build/otb_eval.exe" -ForegroundColor White
Write-Host "  - update_csrt/build/otb_compare.exe" -ForegroundColor White
Write-Host ""
