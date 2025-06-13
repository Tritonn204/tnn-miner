#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

SUDO=
me=$(whoami)
if [[ "$me" != "root" ]]; then
  SUDO=sudo
fi

# macOS section
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected macOS"

  # Check if Homebrew is installed
  if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  # Install LLVM 18
  brew install llvm@18

  # Determine Homebrew prefix (depends on architecture)
  if [[ $(uname -m) == "arm64" ]]; then
    BREW_PREFIX="/opt/homebrew"
  else
    BREW_PREFIX="/usr/local"
  fi

  LLVM_BIN="${BREW_PREFIX}/opt/llvm@18/bin"

  # Add to PATH in shell config
  if [[ -n "$ZSH_VERSION" ]]; then
    SHELL_RC="$HOME/.zshrc"
  else
    SHELL_RC="$HOME/.bash_profile"
  fi

  if ! grep -q "$LLVM_BIN" "$SHELL_RC"; then
    echo "Adding LLVM 18 to PATH in $SHELL_RC"
    echo "export PATH=\"$LLVM_BIN:\$PATH\"" >> "$SHELL_RC"
    export PATH="$LLVM_BIN:$PATH"
  fi

  # Create versioned symlinks for clang, clang++, etc.
  echo "Creating symlinks for versioned LLVM commands..."
  cd "$LLVM_BIN"
  for bin in clang clang++ lld ld.lld; do
    if [[ -f "$bin" && ! -f "${bin}-18" ]]; then
      ln -s "$bin" "${bin}-18"
      echo "  Linked ${bin} -> ${bin}-18"
    fi
  done

  echo "LLVM 18 installed, symlinks created, and PATH updated."

else
  # Linux section
  if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    $SUDO apt update
    $SUDO apt install -y git wget build-essential cmake clang libssl-dev libudns-dev libc++-dev lld libsodium-dev libnuma-dev

    if [[ "$VERSION_CODENAME" == "bookworm" ]]; then
      $SUDO apt install -y libpthread-stubs0-dev
    fi
  fi
fi