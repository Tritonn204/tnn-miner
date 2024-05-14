set sevenzexe="C:\Program Files\7-Zip\7z.exe"
set zstdexe="C:\zstd-v1.5.6-win64\zstd.exe"
if "%1%" == "gh" (
  set sevenzexe=7z
  set zstdexe=zstd
)

mkdir c:\mingw64
mkdir c:\packages
pushd c:\packages
  curl -L https://github.com/brechtsanders/winlibs_mingw/releases/download/14.1.0posix-18.1.5-11.0.1-ucrt-r1/winlibs-x86_64-posix-seh-gcc-14.1.0-llvm-18.1.5-mingw-w64ucrt-11.0.1-r1.zip -o mingw.zip
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-openssl-3.3.0-2-any.pkg.tar.zst -o openssl.tar.zstd
  curl -L https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-fmt-10.2.1-1-any.pkg.tar.zst -o fmt.tar.zstd
popd

%sevenzexe% -y -oc:\ x c:\packages\mingw.zip

%zstdexe% -d -f c:\packages\fmt.tar.zstd
%zstdexe% -d -f c:\packages\openssl.tar.zstd

dir c:\packages\

tar -xf c:\packages\fmt.tar -C c:\
tar -xf c:\packages\openssl.tar -C c:\

xcopy c:\clang64 c:\mingw64\ /E /H /Y /Q

rmdir /s /q c:\clang64\
