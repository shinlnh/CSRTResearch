# Simple build script without presets
param(
    [string]$BuildType = "Release",
    [string]$OpenCVDir = "C:/opencv/build",
    [switch]$Clean,
    [int]$Jobs = 8
)

Write-Host "=== Simple Build Script ===" -ForegroundColor Cyan

# Check OpenCV
if (-not (Test-Path $OpenCVDir)) {
    Write-Host "Error: OpenCV not found at $OpenCVDir" -ForegroundColor Red
    exit 1
}

Write-Host "OpenCV: $OpenCVDir" -ForegroundColor Green
Write-Host "Build: $BuildType" -ForegroundColor White
Write-Host ""

# Clean
if ($Clean) {
    Write-Host "Cleaning..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "pure_csrt_detail\build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "update_csrt\build" -ErrorAction SilentlyContinue
}

# Build Main
Write-Host "=== Main Project ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force "build" | Out-Null
Push-Location "build"

cmake .. -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DOpenCV_DIR=$OpenCVDir `
    -DUSE_CUDA=ON

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}

cmake --build . --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}
Pop-Location
Write-Host "✓ Main project done" -ForegroundColor Green
Write-Host ""

# Build pure_csrt_detail
Write-Host "=== Pure CSRT Detail ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force "pure_csrt_detail\build" | Out-Null
Push-Location "pure_csrt_detail\build"

cmake .. -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DOpenCV_DIR=$OpenCVDir `
    -DUSE_CUDA=ON

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}

cmake --build . --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}
Pop-Location
Write-Host "✓ Pure CSRT Detail done" -ForegroundColor Green
Write-Host ""

# Build update_csrt
Write-Host "=== Update CSRT ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force "update_csrt\build" | Out-Null
Push-Location "update_csrt\build"

cmake .. -G Ninja `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DOpenCV_DIR=$OpenCVDir `
    -DUSE_CUDA=ON

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}

cmake --build . --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit 1
}
Pop-Location
Write-Host "✓ Update CSRT done" -ForegroundColor Green

Write-Host ""
Write-Host "=== ALL DONE ===" -ForegroundColor Green
