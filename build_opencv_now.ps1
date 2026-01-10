# Build OpenCV CUDA with proper Visual Studio environment
Write-Host "=== Building OpenCV with CUDA ===" -ForegroundColor Cyan

# Import Visual Studio environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1"
if (Test-Path $vsPath) {
    & $vsPath -Arch amd64
} else {
    # Try VS 2022
    $vsPath2022 = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
    if (Test-Path $vsPath2022) {
        & $vsPath2022 -Arch amd64
    } else {
        Write-Host "Visual Studio environment not found. Running vcvarsall.bat..." -ForegroundColor Yellow
        & "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    }
}

Write-Host ""
Write-Host "Building OpenCV..." -ForegroundColor Yellow
cmake --build C:\opencv\opencv_build\cuda -j 8

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing OpenCV..." -ForegroundColor Yellow
cmake --install C:\opencv\opencv_build\cuda

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Verify libraries:" -ForegroundColor Cyan
Get-ChildItem "C:\opencv\opencv_build\cuda\lib" -Filter "opencv_tracking*.lib" | Select-Object Name
Get-ChildItem "C:\opencv\opencv_build\cuda\install" -Recurse -Filter "opencv_tracking*.lib" | Select-Object FullName -First 5
