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

# Determine CMake command based on argument
if ($args[0] -eq "ci") {
    & c:\mingw64\bin\cmake.exe -DCMAKE_MAKE_PROGRAM=c:/mingw64/bin/ninja.exe -G "Ninja" ..
} else {
    & c:\mingw64\bin\cmake.exe ..
}

# Run Ninja build
& c:\mingw64\bin\ninja.exe

# Return to original directory
popd
