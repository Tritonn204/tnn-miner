#!/bin/bash

mkdir wincross
pushd wincross
  wget https://github.com/Jake-Shadle/xwin/releases/download/0.6.6-rc.2/xwin-0.6.6-rc.2-x86_64-unknown-linux-musl.tar.gz
  tar -xvf ./xwin-0.6.6-rc.2-x86_64-unknown-linux-musl.tar.gz
  chmod +x ./xwin-0.6.6-rc.2-x86_64-unknown-linux-musl/xwin
  ./xwin-0.6.6-rc.2-x86_64-unknown-linux-musl/xwin --accept-license splat --preserve-ms-arch-notation --output "/src/winsdk"
  ls -la ./xwin-0.6.6-rc.2-x86_64-unknown-linux-musl/

  wget https://ftp.osuosl.org/pub/msys2/mingw/clang64/mingw-w64-clang-x86_64-openssl-3.4.1-1-any.pkg.tar.zst
  tar --zstd -xf mingw-w64-clang-x86_64-openssl-3.4.1-1-any.pkg.tar.zst

  wget https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-boost-1.87.0-2-any.pkg.tar.zst
  tar --zstd -xf mingw-w64-clang-x86_64-boost-1.87.0-2-any.pkg.tar.zst

  wget https://mirror.msys2.org/mingw/clang64/mingw-w64-clang-x86_64-libsodium-1.0.20-1-any.pkg.tar.zst
  tar --zstd -xf mingw-w64-clang-x86_64-libsodium-1.0.20-1-any.pkg.tar.zst
  mkdir build
  pushd build
    cmake -DWIN_CROSS=1 -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 ../../ 
  popd
popd