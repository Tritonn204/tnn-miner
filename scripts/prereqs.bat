@echo OFF
set sevenzexe="C:\Program Files\7-Zip\7z.exe"
REM set zstdexe="C:\zstd-v1.5.6-win64\zstd.exe"
if "%1%" == "ci" (
  set sevenzexe=7z
  REM set zstdexe=zstd
)

dir c:\

mkdir c:\mingw64
mkdir c:\openssl
mkdir c:\packages
pushd c:\packages
  if not exist zstd-v1.5.6-win64.zip (
    curl -L https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-v1.5.6-win64.zip -o zstd-v1.5.6-win64.zip
  )
  if not exist mingw-1410.zip (
    REM curl -L https://github.com/brechtsanders/winlibs_mingw/releases/download/14.1.0posix-18.1.5-11.0.1-ucrt-r1/winlibs-x86_64-posix-seh-gcc-14.1.0-llvm-18.1.5-mingw-w64ucrt-11.0.1-r1.zip -o mingw-1410.zip
    curl -L https://github.com/brechtsanders/winlibs_mingw/releases/download/14.1.0posix-18.1.8-12.0.0-ucrt-r3/winlibs-x86_64-posix-seh-gcc-14.1.0-llvm-18.1.8-mingw-w64ucrt-12.0.0-r3.zip -o mingw-1410.zip
  )
  if not exist openssl-330.tar.zstd (
    curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-openssl-3.3.0-2-any.pkg.tar.zst -o openssl-330.tar.zstd
  )
  if not exist libsodium-1020.tar.zstd (
    curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libsodium-1.0.20-1-any.pkg.tar.zst -o libsodium-1020.tar.zstd
  )
popd

echo Unzip zstd
%sevenzexe% -y -oc:\ x c:\packages\zstd-v1.5.6-win64.zip
set zstdexe="C:\zstd-v1.5.6-win64\zstd.exe"

dir c:\

echo Unzipping MinGW
%sevenzexe% -y -oc:\ x c:\packages\mingw-1410.zip

echo Uncompressing OpenSSL
%zstdexe% -d -f c:\packages\openssl-330.tar.zstd
echo Uncompressing Sodium
%zstdexe% -d -f c:\packages\libsodium-1020.tar.zstd

dir c:\packages\

echo Untarring OpenSSL to c:\openssl
tar -xf c:\packages\openssl-330.tar -C c:\openssl
echo Untarring Sodium
tar -xf c:\packages\libsodium-1020.tar -C c:\
echo Untarring HWLoc
tar -xf c:\packages\hwloc-2111.tar -C c:\

echo Copying from c:\clang64 to c:\mingw64
xcopy c:\clang64 c:\mingw64\ /E /H /Y /Q

rmdir /s /q c:\clang64\
