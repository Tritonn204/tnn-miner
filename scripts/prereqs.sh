#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

LLVM_VER=20
SET_DEFAULT_CLANG="${SET_DEFAULT_CLANG:-0}"   # export SET_DEFAULT_CLANG=1 to make clang -> clang-20 on Linux/mac

SUDO=
me=$(whoami)
if [[ "$me" != "root" ]]; then
  SUDO=sudo
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected macOS"

  if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  echo "Installing LLVM ${LLVM_VER}..."
  brew update
  brew install "llvm@${LLVM_VER}"

  if [[ $(uname -m) == "arm64" ]]; then
    BREW_PREFIX="/opt/homebrew"
  else
    BREW_PREFIX="/usr/local"
  fi

  LLVM_BIN="${BREW_PREFIX}/opt/llvm@${LLVM_VER}/bin"

  # Pick shell rc
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    SHELL_RC="$HOME/.zshrc"
  else
    SHELL_RC="$HOME/.bash_profile"
  fi

  if ! grep -q "$LLVM_BIN" "$SHELL_RC" 2>/dev/null; then
    echo "Adding LLVM ${LLVM_VER} to PATH in $SHELL_RC"
    echo "export PATH=\"$LLVM_BIN:\$PATH\"" >> "$SHELL_RC"
  fi
  export PATH="$LLVM_BIN:$PATH"

  echo "Creating versioned symlinks in ${LLVM_BIN}..."
  cd "$LLVM_BIN"
  for bin in clang clang++ lld ld.lld llvm-ar llvm-ranlib; do
    if [[ -f "$bin" && ! -e "${bin}-${LLVM_VER}" ]]; then
      ln -s "$bin" "${bin}-${LLVM_VER}"
      echo "  Linked ${bin} -> ${bin}-${LLVM_VER}"
    fi
  done

  if [[ "$SET_DEFAULT_CLANG" == "1" ]]; then
    echo "SET_DEFAULT_CLANG=1: using brew llvm@${LLVM_VER} first in PATH (already set)."
    echo "  clang => $(command -v clang)"
    echo "  clang --version => $(clang --version | head -n 1)"
  else
    echo "Not changing your system default clang; use clang-${LLVM_VER} explicitly if needed."
  fi

  echo "Done."

else
  # Linux section
  if [[ -f /etc/os-release ]]; then
    source /etc/os-release

    echo "Detected Linux: ${ID} ${VERSION_CODENAME:-}"

    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release

    # Install LLVM repo helper (supports Debian/Ubuntu)
    echo "Installing LLVM ${LLVM_VER} via apt.llvm.org..."
    curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
    chmod +x /tmp/llvm.sh
    $SUDO /tmp/llvm.sh "${LLVM_VER}"
    rm -f /tmp/llvm.sh

    # Your other deps
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
      git wget build-essential cmake \
      "clang-${LLVM_VER}" "clang++-${LLVM_VER}" \
      "lld-${LLVM_VER}" "llvm-${LLVM_VER}" "llvm-${LLVM_VER}-tools" \
      libssl-dev libudns-dev libc++-dev libsodium-dev libnuma-dev

    if [[ "${VERSION_CODENAME:-}" == "bookworm" ]]; then
      $SUDO apt-get install -y libpthread-stubs0-dev
    fi

    if [[ "$SET_DEFAULT_CLANG" == "1" ]]; then
      echo "SET_DEFAULT_CLANG=1: registering clang-${LLVM_VER} as default via update-alternatives..."
      $SUDO update-alternatives --install /usr/bin/clang clang /usr/bin/clang-"${LLVM_VER}" 100
      $SUDO update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-"${LLVM_VER}" 100
      if [[ -x "/usr/bin/ld.lld-${LLVM_VER}" ]]; then
        $SUDO update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-"${LLVM_VER}" 100
      fi
    else
      echo "Not changing your default clang; use clang-${LLVM_VER} explicitly if needed."
    fi

    echo "clang-${LLVM_VER}: $($(command -v clang-${LLVM_VER}) --version | head -n 1)"
  fi
fi
