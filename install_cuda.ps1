# Download and Install CUDA Toolkit 12.1 (compatible với driver 591.59)

$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe"
$cudaInstaller = "E:\cuda_12.1.0_installer.exe"

Write-Host "Downloading CUDA Toolkit 12.1..." -ForegroundColor Green
Write-Host "URL: $cudaUrl" -ForegroundColor Cyan

try {
    Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaInstaller -UseBasicParsing
    Write-Host "Downloaded to: $cudaInstaller" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installing CUDA Toolkit..." -ForegroundColor Yellow
    Write-Host "This will take 10-20 minutes..." -ForegroundColor Yellow
    
    # Run installer silently
    Start-Process -FilePath $cudaInstaller -ArgumentList "-s" -Wait
    
    Write-Host ""
    Write-Host "CUDA Toolkit installed successfully!" -ForegroundColor Green
    Write-Host "Verifying installation..." -ForegroundColor Cyan
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    nvcc --version
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually from:" -ForegroundColor Yellow
    Write-Host "https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
}
