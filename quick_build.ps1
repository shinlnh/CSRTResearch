# Quick build script without vcpkg dependency
param(
    [string]$BuildType = "Release",
    [string]$OpenCVDir = "C:/opencv/build",
    [switch]$Clean = $false,
    [int]$Jobs = 8
)

Write-Host "=== Quick Build (No vcpkg required) ===" -ForegroundColor Cyan
Write-Host ""

# Check if OpenCV CUDA build exists
if (-not (Test-Path $OpenCVDir)) {
    Write-Host "Error: OpenCV build not found at $OpenCVDir" -ForegroundColor Red
    Write-Host "Please specify correct path: .\quick_build.ps1 -OpenCVDir 'path\to\opencv\build'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using OpenCV from: $OpenCVDir" -ForegroundColor Green
Write-Host "Build Type: $BuildType" -ForegroundColor White
Write-Host "Parallel Jobs: $Jobs" -ForegroundColor White
Write-Host ""

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow
    @("build", "pure_csrt_detail\build", "update_csrt\build") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item -Recurse -Force $_
            Write-Host "  Cleaned: $_" -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# Function to build a project
function Build-Project {
    param(
        [string]$Name,
        [string]$SourceDir,
        [string]$BuildDir
    )
    
    Write-Host "=== Building $Name ===" -ForegroundColor Cyan
    
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }
    
    Push-Location $BuildDir
    
    Write-Host "Configuring..." -ForegroundColor Yellow
    cmake $SourceDir `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DOpenCV_DIR=$OpenCVDir `
        -DUSE_CUDA=ON
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Configuration failed for $Name!" -ForegroundColor Red
        Pop-Location
        return $false
    }
    
    Write-Host "Building..." -ForegroundColor Yellow
    cmake --build . --config $BuildType --parallel $Jobs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed for $Name!" -ForegroundColor Red
        Pop-Location
        return $false
    }
    
    Pop-Location
    Write-Host "✓ $Name built successfully" -ForegroundColor Green
    Write-Host ""
    return $true
}

# Build main project
$success = Build-Project -Name "Main Project" -SourceDir "." -BuildDir "build"
if (-not $success) { exit 1 }

# Build pure_csrt_detail
$success = Build-Project -Name "Pure CSRT Detail" -SourceDir "pure_csrt_detail" -BuildDir "pure_csrt_detail\build"
if (-not $success) { exit 1 }

# Build update_csrt
$success = Build-Project -Name "Update CSRT" -SourceDir "update_csrt" -BuildDir "update_csrt\build"
if (-not $success) { exit 1 }

Write-Host ""
Write-Host "=== All Projects Built Successfully! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Executables:" -ForegroundColor Cyan
Write-Host "  Main:" -ForegroundColor Yellow
Write-Host "    - build\csrtpure.exe" -ForegroundColor White
Write-Host ""
Write-Host "  Pure CSRT Detail:" -ForegroundColor Yellow
Write-Host "    - pure_csrt_detail\build\csrt_demo.exe" -ForegroundColor White
Write-Host "    - pure_csrt_detail\build\otb_eval.exe" -ForegroundColor White
Write-Host "    - pure_csrt_detail\build\otb_compare.exe" -ForegroundColor White
Write-Host "    - pure_csrt_detail\build\test_basketball.exe" -ForegroundColor White
Write-Host ""
Write-Host "  Update CSRT:" -ForegroundColor Yellow
Write-Host "    - update_csrt\build\csrt_demo.exe" -ForegroundColor White
Write-Host "    - update_csrt\build\otb_eval.exe" -ForegroundColor White
Write-Host "    - update_csrt\build\otb_compare.exe" -ForegroundColor White
Write-Host ""

# Check if Ninja is available
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Host "Note: Ninja build system not found in PATH" -ForegroundColor Yellow
    Write-Host "Install it via: winget install Ninja-build.Ninja" -ForegroundColor Yellow
    Write-Host "Or download from: https://github.com/ninja-build/ninja/releases" -ForegroundColor Yellow
}
