# Setup vcpkg for project
param(
    [string]$InstallPath = "C:\vcpkg"
)

Write-Host "=== vcpkg Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if vcpkg already exists
if (Test-Path $InstallPath) {
    Write-Host "vcpkg found at: $InstallPath" -ForegroundColor Green
    
    # Update vcpkg
    Write-Host "Updating vcpkg..." -ForegroundColor Yellow
    Push-Location $InstallPath
    git pull
    .\bootstrap-vcpkg.bat
    Pop-Location
} else {
    Write-Host "Installing vcpkg to: $InstallPath" -ForegroundColor Yellow
    
    # Clone vcpkg
    Write-Host "Cloning vcpkg repository..." -ForegroundColor Yellow
    git clone https://github.com/microsoft/vcpkg.git $InstallPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone vcpkg!" -ForegroundColor Red
        exit 1
    }
    
    # Bootstrap vcpkg
    Write-Host "Bootstrapping vcpkg..." -ForegroundColor Yellow
    Push-Location $InstallPath
    .\bootstrap-vcpkg.bat
    Pop-Location
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to bootstrap vcpkg!" -ForegroundColor Red
        exit 1
    }
}

# Set environment variable for current session
$env:VCPKG_ROOT = $InstallPath
Write-Host ""
Write-Host "VCPKG_ROOT set to: $InstallPath" -ForegroundColor Green

# Add to PATH
$env:PATH = "$InstallPath;$env:PATH"
Write-Host "Added vcpkg to PATH" -ForegroundColor Green

# Integrate vcpkg
Write-Host ""
Write-Host "Integrating vcpkg with system..." -ForegroundColor Yellow
Push-Location $InstallPath
.\vcpkg integrate install
Pop-Location

# Install required packages
Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Yellow
Write-Host "  - Intel TBB (for parallel processing)" -ForegroundColor White

Push-Location $InstallPath
.\vcpkg install tbb:x64-windows
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to install some packages" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To persist VCPKG_ROOT for future sessions, run:" -ForegroundColor Cyan
Write-Host "  [System.Environment]::SetEnvironmentVariable('VCPKG_ROOT', '$InstallPath', 'User')" -ForegroundColor White
Write-Host ""
Write-Host "Or manually add to System Environment Variables:" -ForegroundColor Cyan
Write-Host "  Variable: VCPKG_ROOT" -ForegroundColor White
Write-Host "  Value: $InstallPath" -ForegroundColor White
Write-Host ""

# Ask if user wants to set permanently
$response = Read-Host "Set VCPKG_ROOT permanently for current user? (Y/N)"
if ($response -eq 'Y' -or $response -eq 'y') {
    [System.Environment]::SetEnvironmentVariable('VCPKG_ROOT', $InstallPath, 'User')
    Write-Host "VCPKG_ROOT set permanently!" -ForegroundColor Green
}

Write-Host ""
Write-Host "You can now run: .\build_project.ps1" -ForegroundColor Cyan
