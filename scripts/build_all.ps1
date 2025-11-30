param(
    [string]$TnnVersion,
    [string]$Target
)

# If script is called without named params, fall back to $args
if (-not $TnnVersion -and $args.Count -ge 1) { $TnnVersion = $args[0] }
if (-not $Target -and $args.Count -ge 2) { $Target = $args[1] }

if (-not $TnnVersion) {
    Write-Host "Usage: .\build_all.ps1 <TNN_VERSION> [TARGET]"
    exit 1
}

$TNN_VERSION   = $TnnVersion
$targetToBuild = if ($Target) { $Target.ToLower() } else { "" }

# Paths relative to this script
$ScriptRoot = $PSScriptRoot  # folder containing build_all.ps1
$RepoRoot   = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
$LogRoot    = Join-Path $RepoRoot "build-logs"

if (-not (Test-Path $LogRoot)) {
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
}

Write-Host "Repo root: $RepoRoot"
Write-Host "TNN_VERSION: $TNN_VERSION"
Write-Host "Target to build: $targetToBuild"

function Get-ProcessorCount {
    $count = (Get-WmiObject Win32_Processor).NumberOfLogicalProcessors
    Write-Host "Processor Count: $count"
    return $count
}

function Invoke-LoggedCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$LogFile
    )

    Write-Host ">> $FilePath $($Arguments -join ' ')"
    Write-Host "   (logging to $LogFile)"

    # Ensure log directory exists
    $logDir = Split-Path $LogFile -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    # Clear existing log
    if (Test-Path $LogFile) {
        Remove-Item $LogFile -Force
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = [string]::Join(' ', $Arguments)
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi

    $null = $proc.Start()

    $logStream = [System.IO.StreamWriter]::new($LogFile, $true)

    try {
        while (-not $proc.HasExited -or
               -not $proc.StandardOutput.EndOfStream -or
               -not $proc.StandardError.EndOfStream) {

            # Stdout
            while (-not $proc.StandardOutput.EndOfStream) {
                $line = $proc.StandardOutput.ReadLine()
                if ($null -ne $line) {
                    Write-Host $line
                    $logStream.WriteLine($line)
                }
            }

            # Stderr
            while (-not $proc.StandardError.EndOfStream) {
                $line = $proc.StandardError.ReadLine()
                if ($null -ne $line) {
                    Write-Host $line
                    $logStream.WriteLine($line)
                }
            }

            Start-Sleep -Milliseconds 10
        }

        $proc.WaitForExit()
        return $proc.ExitCode
    }
    finally {
        $logStream.Flush()
        $logStream.Dispose()
        $proc.Dispose()
    }
}

function Get-NvccPath {
    # 1) Try PATH first
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        Write-Host "Found nvcc via PATH: $($cmd.Source)"
        return $cmd.Source
    }

    # 2) Try standard CUDA location
    $defaultRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $defaultRoot) {
        $versions = Get-ChildItem $defaultRoot -Directory |
            Sort-Object Name -Descending

        foreach ($v in $versions) {
            $candidate = Join-Path $v.FullName "bin\nvcc.exe"
            if (Test-Path $candidate) {
                Write-Host "Found nvcc via CUDA root: $candidate"
                return $candidate
            }
        }
    }

    Write-Host "nvcc.exe not found in PATH or default CUDA location."
    return $null
}

function Build-Target {
    param (
        [string]$TargetDir,
        [string]$HipFlag,      # "ON" or "OFF"
        [string]$HipPlatform   # "amd", "nvidia", or ""
    )

    Write-Host "==============================================="
    Write-Host "Building target in directory: '$TargetDir'"
    Write-Host "HIP Flag: $HipFlag"
    Write-Host "HIP Platform: $HipPlatform"
    Write-Host "==============================================="

    $buildDir = Join-Path $RepoRoot ("hip-build\win32\" + $TargetDir)

    # Ensure build directory exists
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    }

    # Remove CMakeCache.txt to refresh cache for each target
    $CacheFile = Join-Path $buildDir "CMakeCache.txt"
    if (Test-Path $CacheFile) {
        Write-Host "Removing existing CMake cache: $CacheFile"
        Remove-Item $CacheFile -Force
    }

    $hip_path_quoted = "`"$env:HIP_PATH`""
    Write-Host "HIP_PATH: $hip_path_quoted"

    # If building for NVIDIA HIP, locate nvcc
    $nvccPath = $null
    if ($HipFlag -eq "ON" -and $HipPlatform -eq "nvidia") {
        $nvccPath = Get-NvccPath
        if (-not $nvccPath) {
            Write-Host "ERROR: Could not find nvcc.exe for NVIDIA HIP build."
            return $false
        } else {
            Write-Host "Using nvcc for HIP compiler: $nvccPath"
        }
    }

    $cmakeCommand = "cmake"

    # ---------- CONFIGURE STEP ----------
    $configureLog = Join-Path $LogRoot "cmake_configure_${HipPlatform}_${TargetDir}.log"

    $cmakeArgs = @(
        "-S", $RepoRoot,
        "-B", $buildDir,
        "-G", "Ninja",
        "-DCMAKE_HIP_PLATFORM=$HipPlatform",
        "-DWITH_HIP=$HipFlag",
        "-DHIP_PLATFORM=$HipPlatform",
        "-DTNN_VERSION=$TNN_VERSION",
        "--fresh"
    )

    if ($env:HIP_PATH) {
        $cmakeArgs += "-DHIP_PATH=""$($env:HIP_PATH)"""
        # Pre-seed ROCm root so CMakeDetermineHIPCompiler.cmake
        # doesn't fail on hipconfig --rocmpath when HIP_PLATFORM=nvidia.
        $cmakeArgs += "-DCMAKE_HIP_COMPILER_ROCM_ROOT=""$($env:HIP_PATH)"""

        # Let CMake find hip-lang (and other HIP packages) under the ROCm tree
        $cmakeArgs += "-DCMAKE_PREFIX_PATH=""$($env:HIP_PATH)"""

        # Optional but robust: point directly at hip-lang-config.cmake if it exists
        $hipLangDir = Join-Path $env:HIP_PATH "lib\cmake\hip-lang"
        if (Test-Path $hipLangDir) {
            $cmakeArgs += "-Dhip-lang_DIR=""$hipLangDir"""
        }
    }

    if ($nvccPath) {
        # Tell CMake explicitly which HIP compiler to use for NVIDIA backend
        $cmakeArgs += "-DCMAKE_HIP_COMPILER=""$nvccPath"""
    }

    Write-Host ""
    Write-Host "---- CMake configure ----"
    $configureExitCode = Invoke-LoggedCommand -FilePath $cmakeCommand -Arguments $cmakeArgs -LogFile $configureLog

    if ($configureExitCode -ne 0) {
        Write-Host ""
        Write-Host "CMake configure FAILED for platform '$HipPlatform', target dir '$TargetDir'."
        Write-Host "Exit code: $configureExitCode"
        Write-Host "See log: $configureLog"
        return $false
    }

    # ---------- BUILD STEP ----------
    $procCount = Get-ProcessorCount
    $buildLog  = Join-Path $LogRoot "cmake_build_${HipPlatform}_${TargetDir}.log"

    $buildArgs = @(
        "--build", $buildDir,
        "--target", "all",
        "--parallel", $procCount
    )

    Write-Host ""
    Write-Host "---- CMake build ----"
    $buildExitCode = Invoke-LoggedCommand -FilePath $cmakeCommand -Arguments $buildArgs -LogFile $buildLog

    if ($buildExitCode -ne 0) {
        Write-Host ""
        Write-Host "Build FAILED for platform '$HipPlatform', target dir '$TargetDir'."
        Write-Host "Exit code: $buildExitCode"
        Write-Host "See log: $buildLog"
        return $false
    }

    Write-Host ""
    Write-Host "Build SUCCEEDED for platform '$HipPlatform', target dir '$TargetDir'."
    Write-Host "Configure log: $configureLog"
    Write-Host "Build log:     $buildLog"
    return $true
}

# Save original PATH so we can restore it later
$originalPath = $env:PATH

try {
    # Temporarily remove any mingw64 paths from PATH
    $env:PATH = ($env:PATH -split ';' |
        Where-Object {
            $_ -and ($_ -notmatch '(?i)\\mingw64(\\|$)') -and ($_ -notmatch '(?i)/mingw64(/|$)')
        }
    ) -join ';'

    Write-Host "PATH filtered to exclude mingw64 for this build."

    # Get HIP path and handle spaces in the path
    $env:HIP_PATH = (& hipconfig --path).Trim()
    Write-Host "HIP_PATH: $env:HIP_PATH"

    # Build for AMD using ROCm
    if ($targetToBuild -eq "" -or $targetToBuild -eq "amd") {
        if (-not (Build-Target "amd" "ON" "amd")) {
            Write-Host "Failed to build for AMD."
        }
    }

    # Build for NVIDIA using HIP (HIP_PLATFORM=nvidia)
    if ($targetToBuild -eq "" -or $targetToBuild -eq "nvidia") {
        if (-not (Build-Target "nvidia" "ON" "nvidia")) {
            Write-Host "Failed to build for NVIDIA."
        }
    }

    # Build without HIP (CPU-only)
    if ($targetToBuild -eq "" -or $targetToBuild -eq "cpu") {
        if (-not (Build-Target "cpu" "OFF" "")) {
            Write-Host "Failed to build for CPU-only."
        }
    }
}
finally {
    # Always restore original PATH
    $env:PATH = $originalPath
    Write-Host "PATH restored to original value."
}
