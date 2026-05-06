 param (
    [string]$version = "",
    [switch]$clangcl
 )

function Import-MsvcEnvironment {
    param(
        [string]$Architecture = "x64"
    )

    $vswhereDefault = Join-Path "${env:ProgramFiles(x86)}" "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhereDefault)) {
        Write-Host "vswhere.exe not found at $vswhereDefault. Make sure Visual Studio with C++ tools is installed."
        return $false
    }

    $vsInstallPath = & $vswhereDefault `
        -latest `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath `
        -nologo

    if (-not $vsInstallPath) {
        Write-Host "No Visual Studio installation with C++ tools found."
        return $false
    }

    $vcvarsall = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        Write-Host "vcvarsall.bat not found under $vsInstallPath."
        return $false
    }

    Write-Host "Using Visual Studio at: $vsInstallPath"
    Write-Host "Running vcvarsall.bat $Architecture..."

    $cmdOutput = & cmd.exe /c "`"$vcvarsall`" $Architecture >nul && set" 2>$null

    if (-not $cmdOutput) {
        Write-Host "Failed to run vcvarsall.bat or capture environment."
        return $false
    }

    foreach ($line in $cmdOutput) {
        $pair = $line -split "=", 2
        if ($pair.Length -eq 2) {
            Set-Item -Path "Env:$($pair[0])" -Value $pair[1]
        }
    }

    Write-Host "MSVC environment imported (PATH, INCLUDE, LIB, etc.)."
    return $true
}

function Find-ClangCl {
    $clangClExe = (Get-Command clang-cl -ErrorAction SilentlyContinue).Source
    if (-not $clangClExe) {
        $clangClExe = "C:\Program Files\LLVM\bin\clang-cl.exe"
        if (-not (Test-Path $clangClExe)) {
            Write-Host "ERROR: clang-cl.exe not found in PATH or default LLVM location."
            return $null
        }
    }
    return $clangClExe
}

function Setup-MsvcClangCl {
    # Import MSVC environment for Windows SDK + CRT lib paths
    if (-not (Import-MsvcEnvironment -Architecture "x64")) {
        Write-Host "ERROR: Could not import MSVC environment. Build requires Visual Studio with C++ tools."
        return $false
    }

    # Strip MinGW from PATH to avoid picking up MinGW's lld-link, libs, etc.
    $env:PATH = ($env:PATH -split ';' |
        Where-Object {
            $_ -and ($_ -notmatch '(?i)\\mingw64(\\|$)') -and ($_ -notmatch '(?i)/mingw64(/|$)')
        }
    ) -join ';'
    Write-Host "PATH filtered to exclude mingw64."
    return $true
}

# Build directory
if ($clangcl) {
    $subDir = "clangcl"
} else {
    $subDir = "mingw"
}
$buildDir = "build\$subDir"

if (!(Test-Path .\$buildDir)) {
    New-Item -ItemType Directory -Path .\$buildDir -Force
}

pushd .\$buildDir

set-variable -name "VER_SETTING" -Value ""
if (-not $version -eq "") {
  set-variable -name "VER_SETTING" -value "-DTNN_VERSION=$version"
}

if ($clangcl) {
    Write-Host "=== clang-cl (MSVC ABI) CPU build ==="

    if (-not (Setup-MsvcClangCl)) {
        popd
        exit 1
    }

    $clangClExe = Find-ClangCl
    if (-not $clangClExe) {
        popd
        exit 1
    }
    Write-Host "Using clang-cl: $clangClExe"

    & cmake $VER_SETTING -Wno-dev `
        -DWITH_HIP=OFF `
        -DUSE_CLANG_CL=ON `
        -DUSE_ASTRO_SPSA=OFF `
        "-DCMAKE_C_COMPILER=$clangClExe" `
        "-DCMAKE_CXX_COMPILER=$clangClExe" `
        -G "Ninja" ..\..

} else {
    # Original MinGW/Clang flow
    if (!(Get-Command clang -ErrorAction SilentlyContinue)) {
        $env:PATH = "c:\mingw64\bin;" + $env:PATH
    }

    if (!(Get-Command clang -ErrorAction SilentlyContinue)) {
        Get-ChildItem c:\mingw64\bin\
        Write-Output "Build will fail. Cannot find clang.exe"
    }

    & cmake $VER_SETTING -Wno-dev -DWITH_HIP=OFF -G "Ninja" ..\..
}

# Run Ninja build
& ninja -j (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors

popd
