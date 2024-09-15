# Function to run cmake and build only if cmake succeeds
function Build-Target {
    param (
        [string]$TargetDir,
        [string]$HipFlag,
        [string]$HipPlatform,
        [string]$TNN_VERSION
    )

    # Print parameters for debugging
    Write-Host "Building target in directory: $TargetDir"
    Write-Host "HIP Flag: $HipFlag"
    Write-Host "HIP Platform: $HipPlatform"
    Write-Host "TNN Version: $TNN_VERSION"

    # Ensure HIP platform environment variable is set
    $env:HIP_PLATFORM = $HipPlatform
    $hip_path_quoted = "`"$env:HIP_PATH`""

    # Define CMake arguments
    $cmakeCommand = "cmake"
    $cmakeArgs = @(
        "-S", "..",
        "-B", "../bin/win32/$TargetDir",
        "-DCMAKE_PREFIX_PATH=$hip_path_quoted",
        "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$hip_path_quoted"
        "-DCMAKE_HIP_PLATFORM=$HipPlatform",
        "-DWITH_HIP=$HipFlag",
        "-DTNN_VERSION=$TNN_VERSION",
        "--fresh"
    )

    # Output the command being run
    Write-Host "Running CMake with args: $cmakeArgs"

    # Run the cmake command and capture both the standard output and standard error
    $outputFile = "cmake_output.txt"
    $errorFile = "cmake_error.txt"

    $process = Start-Process -FilePath $cmakeCommand -ArgumentList $cmakeArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput $outputFile `
        -RedirectStandardError $errorFile -Wait

    # Output the result of the process
    if ($process.ExitCode -eq 0) {
        Write-Host "CMake command executed successfully."
    } else {
        Write-Host "CMake command failed with exit code: $($process.ExitCode)"
    }

    # Display the output and error files
    Write-Host "`nCMake Output:"
    Get-Content $outputFile

    Write-Host "`nCMake Errors:"
    Get-Content $errorFile
}

# Example usage of the function:
$env:HIP_PATH = (& hipconfig --path).Trim()
# $env:ROCM_PATH = (& hipconfig --rocmpath).Trim()

# Run build for AMD (with HIP)
Build-Target "" "ON" "amd" "1.0.0"

# # Run build for NVIDIA (with HIP)
# Build-Target "nvidia_build" "ON" "nvidia" "1.0.0"

# Run build for CPU-only (without HIP)
Build-Target "" "OFF" "" "1.0.0"
