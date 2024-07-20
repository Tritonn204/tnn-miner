 param (
    [string]$version = ""
 )

# Create build directory
if (!(Test-Path .\build)) {
    New-Item -ItemType Directory -Path .\build
}

pushd .\build

# Check if clang is in PATH, otherwise add it
if (!(Get-Command clang -ErrorAction SilentlyContinue)) {
    $env:PATH = "c:\mingw64\bin;" + $env:PATH
}

# Check again if clang is in PATH, print directory contents if not found
if (!(Get-Command clang -ErrorAction SilentlyContinue)) {
    Get-ChildItem c:\mingw64\bin\
    Write-Output "Build will fail. Cannot find clang.exe"
}

set-variable -name "VER_SETTING" -Value ""
if (-not $version -eq "") {
  set-variable -name "VER_SETTING" -value "-DTNN_VERSION=$version"
}

# Determine CMake command based on argument
#& c:\mingw64\bin\cmake.exe $VER_SETTING -DCMAKE_MAKE_PROGRAM=c:/mingw64/bin/ninja.exe --debug-trycompile -G "Ninja" ..
& c:\mingw64\bin\cmake.exe $VER_SETTING --debug-trycompile --trace-expand --trace-redirect=./asdf.txt -G "Ninja" ..

# Run Ninja build
& c:\mingw64\bin\ninja.exe

# Return to original directory
popd
