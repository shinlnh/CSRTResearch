# Build OpenCV with CUDA using vcpkg

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Building OpenCV with CUDA Support" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This process will take 2-4 hours!" -ForegroundColor Yellow
Write-Host "GPU: RTX 5060 (Compute Capability 8.9)" -ForegroundColor Cyan
Write-Host ""

# Navigate to vcpkg
Set-Location E:\vcpkg

# Set environment
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"

Write-Host "Step 1: Installing OpenCV with CUDA, cuDNN, and contrib modules..." -ForegroundColor Green

# Install OpenCV with CUDA
.\vcpkg install opencv[core,cuda,cudnn,contrib,dnn,ffmpeg]:x64-windows

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "OpenCV with CUDA installed successfully!" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Integrate with system
Write-Host "Step 2: Integrating vcpkg with Visual Studio..." -ForegroundColor Green
.\vcpkg integrate install

Write-Host ""
Write-Host "DONE! Now you can rebuild your project with CUDA support." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd E:\Programming\C\C2P\Project\CSRTResearch\update_csrt" -ForegroundColor Cyan
Write-Host "2. mkdir build_cuda" -ForegroundColor Cyan
Write-Host "3. cd build_cuda" -ForegroundColor Cyan
Write-Host "4. cmake .. -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_TOOLCHAIN_FILE=E:/vcpkg/scripts/buildsystems/vcpkg.cmake" -ForegroundColor Cyan
Write-Host "5. cmake --build . --config Release" -ForegroundColor Cyan
