# Verify CUDA Installation

Write-Host "Verifying CUDA Installation..." -ForegroundColor Green
Write-Host ""

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check nvcc
Write-Host "1. Checking nvcc compiler..." -ForegroundColor Cyan
try {
    $nvccVersion = nvcc --version 2>&1
    Write-Host $nvccVersion -ForegroundColor Green
} catch {
    Write-Host "nvcc not found. Please restart PowerShell or add CUDA to PATH manually:" -ForegroundColor Red
    Write-Host "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "2. Checking CUDA environment variables..." -ForegroundColor Cyan
$cudaPath = [System.Environment]::GetEnvironmentVariable("CUDA_PATH","Machine")
if ($cudaPath) {
    Write-Host "CUDA_PATH = $cudaPath" -ForegroundColor Green
} else {
    Write-Host "CUDA_PATH not set" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Checking GPU..." -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe") {
    Write-Host "✅ CUDA Toolkit installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Build OpenCV with CUDA" -ForegroundColor Yellow
    Write-Host "Run: .\build_opencv_cuda.ps1" -ForegroundColor Cyan
} else {
    Write-Host "❌ CUDA Toolkit installation incomplete" -ForegroundColor Red
    Write-Host "Please complete the CUDA installer" -ForegroundColor Yellow
}
Write-Host "===================================================================" -ForegroundColor Cyan
