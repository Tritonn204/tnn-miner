@echo OFF
setlocal

set sevenzexe="C:\Program Files\7-Zip\7z.exe"
set MODE=%1
set LEAN=0

if /I "%MODE%"=="ci" (
  set sevenzexe=7z
)

if /I "%MODE%"=="lean" (
  set LEAN=1
)

set PKGROOT=C:\packages
set TOOLROOT=C:\mingw64
set STAGE=C:\pkgstage

mkdir %PKGROOT% 2>nul
mkdir %TOOLROOT% 2>nul
mkdir %STAGE% 2>nul

pushd %PKGROOT%

if %LEAN%==0 (
  if not exist zstd-v1.5.7-win64.zip (
    curl -L https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-v1.5.7-win64.zip -o zstd-v1.5.7-win64.zip || exit /b 1
  )
)

REM ── Base toolchain: winlibs GCC 14.2.0 + LLVM 18 + MinGW sysroot ──
REM This bundle includes a statically-linked lld, so no CRT mismatch issues.
if not exist mingw-1420.zip (
  curl -L https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-18.1.8-12.0.0-ucrt-r1/winlibs-x86_64-posix-seh-gcc-14.2.0-llvm-18.1.8-mingw-w64ucrt-12.0.0-r1.zip -o mingw-1420.zip || exit /b 1
)

REM ── LLVM/Clang 20 overlay (replaces the bundled Clang 18) ──
if not exist mingw-w64-x86_64-clang-20-20.1.8-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/mingw64/mingw-w64-x86_64-clang-20-20.1.8-1-any.pkg.tar.zst -o mingw-w64-x86_64-clang-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-x86_64-llvm-20-20.1.8-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/mingw64/mingw-w64-x86_64-llvm-20-20.1.8-1-any.pkg.tar.zst -o mingw-w64-x86_64-llvm-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-x86_64-compiler-rt-20-20.1.8-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/mingw64/mingw-w64-x86_64-compiler-rt-20-20.1.8-1-any.pkg.tar.zst -o mingw-w64-x86_64-compiler-rt-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-x86_64-zstd-1.5.7-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/mingw64/mingw-w64-x86_64-zstd-1.5.7-1-any.pkg.tar.zst -o mingw-w64-x86_64-zstd-1.5.7-1-any.pkg.tar.zst || exit /b 1
)

REM ── LLD 21 from MSYS2 clang64 repo (UCRT-based, compatible with winlibs) ──
if not exist mingw-w64-clang-x86_64-lld-21.1.8-4-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-lld-21.1.8-4-any.pkg.tar.zst -o mingw-w64-clang-x86_64-lld-21.1.8-4-any.pkg.tar.zst || exit /b 1
)

REM ── LLD runtime dependencies (all clang64/UCRT) ──
if not exist mingw-w64-clang-x86_64-llvm-libs-21.1.8-4-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-llvm-libs-21.1.8-4-any.pkg.tar.zst -o mingw-w64-clang-x86_64-llvm-libs-21.1.8-4-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-libc++-21.1.8-1-any.pkg.tar.zst (
  curl -L "https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libc++-21.1.8-1-any.pkg.tar.zst" -o mingw-w64-clang-x86_64-libc++-21.1.8-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-libunwind-21.1.8-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libunwind-21.1.8-1-any.pkg.tar.zst -o mingw-w64-clang-x86_64-libunwind-21.1.8-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-libffi-3.5.2-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libffi-3.5.2-1-any.pkg.tar.zst -o mingw-w64-clang-x86_64-libffi-3.5.2-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-libxml2-2.15.2-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libxml2-2.15.2-1-any.pkg.tar.zst -o mingw-w64-clang-x86_64-libxml2-2.15.2-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-libiconv-1.18-1-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libiconv-1.18-1-any.pkg.tar.zst -o mingw-w64-clang-x86_64-libiconv-1.18-1-any.pkg.tar.zst || exit /b 1
)

if not exist mingw-w64-clang-x86_64-zlib-1.3.2-2-any.pkg.tar.zst (
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-zlib-1.3.2-2-any.pkg.tar.zst -o mingw-w64-clang-x86_64-zlib-1.3.2-2-any.pkg.tar.zst || exit /b 1
)

if %LEAN%==0 (
  if not exist openssl-341.tar.zstd (
    curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-openssl-3.4.1-1-any.pkg.tar.zst -o openssl-341.tar.zstd || exit /b 1
  )

  if not exist libsodium-1020.tar.zstd (
    curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libsodium-1.0.20-1-any.pkg.tar.zst -o libsodium-1020.tar.zstd || exit /b 1
  )
)

popd

REM ── Install zstd bootstrap ──
if %LEAN%==0 (
  echo Installing zstd bootstrap
  %sevenzexe% -y -oC:\ x %PKGROOT%\zstd-v1.5.7-win64.zip || exit /b 1
  set zstdexe="C:\zstd-v1.5.7-win64\zstd.exe"
) else (
  set zstdexe=zstd
)

REM ── Install winlibs base (extracts as mingw64\ at C:\) ──
echo.
echo Installing winlibs base toolchain (GCC 14.2.0 + LLVM 18.1.8)
%sevenzexe% -y -oC:\ x %PKGROOT%\mingw-1420.zip || exit /b 1

REM ── Overlay MSYS2 Clang 20 packages on top ──
call :install_pkg mingw-w64-x86_64-clang-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-x86_64-llvm-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-x86_64-compiler-rt-20-20.1.8-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-x86_64-zstd-1.5.7-1-any.pkg.tar.zst || exit /b 1

REM ── Overlay LLD 21 + deps from clang64 (UCRT) ──
call :install_pkg mingw-w64-clang-x86_64-libunwind-21.1.8-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-libc++-21.1.8-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-libiconv-1.18-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-zlib-1.3.2-2-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-libffi-3.5.2-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-libxml2-2.15.2-1-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-llvm-libs-21.1.8-4-any.pkg.tar.zst || exit /b 1
call :install_pkg mingw-w64-clang-x86_64-lld-21.1.8-4-any.pkg.tar.zst || exit /b 1

if %LEAN%==0 (
  echo Uncompressing OpenSSL
  %zstdexe% -d -f %PKGROOT%\openssl-341.tar.zstd
  echo Untarring OpenSSL to c:\openssl
  mkdir c:\openssl 2>nul
  tar -xf %PKGROOT%\openssl-341.tar -C c:\openssl

  echo Uncompressing Sodium
  %zstdexe% -d -f %PKGROOT%\libsodium-1020.tar.zstd
  echo Untarring Sodium to c:\sodium
  mkdir c:\sodium 2>nul
  tar -xf %PKGROOT%\libsodium-1020.tar -C c:\sodium
)

if not exist %TOOLROOT%\bin\clang++.exe (
  echo ERROR: clang++.exe not found in %TOOLROOT%\bin
  echo Searching for it...
  where /r %TOOLROOT% clang++.exe
  exit /b 1
)

echo.
echo Done.
echo Verify:
echo   %TOOLROOT%\bin\clang++ --version
echo   %TOOLROOT%\bin\ld.lld --version
exit /b 0

:install_pkg
set PKGFILE=%~1
set TARFILE=%PKGROOT%\%~n1

echo.
echo Processing %PKGFILE%

if not exist %PKGROOT%\%PKGFILE% (
  echo ERROR: Missing %PKGROOT%\%PKGFILE%
  exit /b 1
)

%zstdexe% -d -f %PKGROOT%\%PKGFILE%
if errorlevel 1 exit /b 1

del /f /q %PKGROOT%\%PKGFILE%
if errorlevel 1 (
  echo WARNING: failed to delete %PKGROOT%\%PKGFILE%
)

rmdir /s /q %STAGE% 2>nul
mkdir %STAGE%

tar -xf %TARFILE% -C %STAGE%
if errorlevel 1 exit /b 1

del /f /q %TARFILE%
if errorlevel 1 (
  echo WARNING: failed to delete %TARFILE%
)

if exist %STAGE%\mingw64\opt\llvm-20 (
  echo Copying from mingw64\opt\llvm-20 to %TOOLROOT%
  xcopy %STAGE%\mingw64\opt\llvm-20\* %TOOLROOT%\ /E /H /Y /Q >nul
  if errorlevel 1 exit /b 1
) else if exist %STAGE%\opt\llvm-20 (
  echo Copying from opt\llvm-20 to %TOOLROOT%
  xcopy %STAGE%\opt\llvm-20\* %TOOLROOT%\ /E /H /Y /Q >nul
  if errorlevel 1 exit /b 1
) else if exist %STAGE%\clang64 (
  echo Copying from clang64 to %TOOLROOT%
  xcopy %STAGE%\clang64\* %TOOLROOT%\ /E /H /Y /Q >nul
  if errorlevel 1 exit /b 1
) else if exist %STAGE%\mingw64 (
  echo Copying from mingw64 to %TOOLROOT%
  xcopy %STAGE%\mingw64\* %TOOLROOT%\ /E /H /Y /Q >nul
  if errorlevel 1 exit /b 1
) else (
  echo ERROR: no recognized layout found in %STAGE%
  dir %STAGE%
  exit /b 1
)

exit /b 0
