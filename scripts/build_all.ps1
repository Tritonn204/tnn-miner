# Check if TNN_VERSION is passed as an argument
if (-not $args[0]) {
    Write-Host "Usage: .\build.ps1 <TNN_VERSION>"
    exit 1
}

$TNN_VERSION = $args[0]

# Print TNN_VERSION for debugging
Write-Host "TNN_VERSION: $TNN_VERSION"

# Function to run cmake and build only if cmake succeeds
function Build-Target {
    param (
        [string]$TargetDir,
        [string]$HipFlag,
        [string]$HipPlatform
    )

    # Print parameters for debugging
    Write-Host "Building target in directory: $TargetDir"
    Write-Host "HIP Flag: $HipFlag"
    Write-Host "HIP Platform: $HipPlatform"

    # Remove CMakeCache.txt to refresh cache for each target
    $CacheFile = "../bin/win32/$TargetDir/CMakeCache.txt"
    if (Test-Path $CacheFile) {
        Remove-Item $CacheFile
    }

    # Set HIP platform environment variable
    $env:HIP_PLATFORM = $HipPlatform

    # Print the HIP_PATH for debugging
    $hip_path_quoted = "`"$env:HIP_PATH`""
    Write-Host "HIP_PATH: $hip_path_quoted"

    # Define file paths for output and error logs
    $outputFile = "cmake_output.txt"
    
    # Command to run cmake
    $cmakeCommand = "cmake"
    $cmakeArgs = @(
        "-S", "..",
        "-B", "../bin/win32/$TargetDir",
        # "-DCMAKE_PREFIX_PATH=$hip_path_quoted",
        "-DCMAKE_HIP_PLATFORM=$HipPlatform",
        "-DWITH_HIP=$HipFlag",
        "-DHIP_PLATFORM=$HipPlatform",
        "-DTNN_VERSION=$TNN_VERSION",
        "--log-level=STATUS"
    )

    # Run cmake command with real-time output
    $process = Start-Process -FilePath $cmakeCommand -ArgumentList $cmakeArgs -NoNewWindow -PassThru  -RedirectStandardError $outputFile

    Get-Content $outputFile -Wait

    if ($process.ExitCode -ne 0) {
        Write-Host "CMake failed, skipping build."
        return $false
    }

    # Build command
    $buildArgs = @(
        "--build", "../bin/win32/$TargetDir",
        "--target", "all",
        "--verbose"
    )

    # Run the build command with real-time output
    $process = Start-Process -FilePath $cmakeCommand -ArgumentList $buildArgs -NoNewWindow -PassThru -RedirectStandardError $outputFile

    Get-Content $outputFile -Wait

    if ($process.ExitCode -ne 0) {
        Write-Host "Build failed"
        return $false
    }

    return $true
}

# Function to get processor count for parallel build
function Get-ProcessorCount {
    $count = (Get-WmiObject Win32_Processor).NumberOfLogicalProcessors
    Write-Host "Processor Count: $count"
    return $count
}

# Get HIP and ROCm paths and handle spaces in the paths
$env:HIP_PATH = (& hipconfig --path).Trim()
$env:ROCM_PATH = (& hipconfig --rocmpath).Trim()

# Print HIP and ROCm paths for debugging
Write-Host "HIP_PATH: $env:HIP_PATH"
Write-Host "ROCM_PATH: $env:ROCM_PATH"

# Build for AMD using ROCm
if (-not (Build-Target "" "ON" "amd")) {
    Write-Host "Failed to build for AMD."
}

# Build for NVIDIA using hipcc (with HIP_PLATFORM=nvidia)
if (-not (Build-Target "" "ON" "nvidia")) {
    Write-Host "Failed to build for NVIDIA."
}

# Build without HIP
if (-not (Build-Target "" "OFF" "")) {
    Write-Host "Failed to build for CPU-only."
}
