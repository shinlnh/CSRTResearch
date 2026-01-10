# Script tải và setup OpenCV CUDA cho Windows

# Download OpenCV with CUDA (prebuilt)
$opencvVersion = "4.8.0"
$downloadUrl = "https://github.com/opencv/opencv/releases/download/$opencvVersion/opencv-$opencvVersion-windows.exe"
$outputPath = "E:\opencv-$opencvVersion-windows.exe"
$installPath = "E:\opencv"

Write-Host "Downloading OpenCV $opencvVersion..." -ForegroundColor Green
Invoke-WebRequest -Uri $downloadUrl -OutFile $outputPath

Write-Host "Downloaded to: $outputPath" -ForegroundColor Green
Write-Host "Run the installer to extract to: $installPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "NOTE: The prebuilt OpenCV does NOT include CUDA modules." -ForegroundColor Red
Write-Host "To get CUDA support, you need to build from source with Visual Studio." -ForegroundColor Yellow
