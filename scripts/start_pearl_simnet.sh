#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./start_pearl_simnet.sh create-wallet
#   ./start_pearl_simnet.sh wallet
#   ./start_pearl_simnet.sh address
#   ./start_pearl_simnet.sh node <mining-address>
#   ./start_pearl_simnet.sh gateway <mining-address>
#   ./start_pearl_simnet.sh all [mining-address]   # optional: provide address, or auto-fetch from wallet

STAGE="${1:-}"
MINING_ADDRESS="${2:-}"

PEARL_ROOT="${PEARL_ROOT:-$HOME/pearl}"
RPC_USER="${RPC_USER:-rpcuser}"
RPC_PASS="${RPC_PASS:-rpcpass}"
WALLET_STARTUP_SECONDS="${WALLET_STARTUP_SECONDS:-4}"
WALLET_DATA_DIR="${WALLET_DATA_DIR:-/tmp/pearl-simnet-wallet}"

BIN_DIR="$PEARL_ROOT/bin"
OYSTER="$BIN_DIR/oyster"
PRLCTL="$BIN_DIR/prlctl"
PEARLD="$BIN_DIR/pearld"

die() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[pearl-simnet] $*"; }

require_file() {
  [ -f "$1" ] || die "$2 not found: $1"
}

require_dir() {
  [ -d "$1" ] || die "$2 not found: $1"
}

require_mining_address() {
  [ -n "$MINING_ADDRESS" ] || die "Stage '$STAGE' requires a mining address as second argument."
}

require_dir  "$PEARL_ROOT" "Pearl root"
require_dir  "$BIN_DIR"    "Pearl bin dir"

launch_bg() {
  local name="$1"
  shift
  info "launching $name in background: $*"
  nohup "$@" > "/tmp/pearl-$name.log" 2>&1 &
  local pid=$!
  info "$name PID: $pid"
  # Save pid under the variable name "name_PID" (e.g. node_PID, wallet_PID)
  eval "${name}_PID=$pid"
}

get_mining_address_from_wallet() {
  require_file "$PRLCTL" "prlctl"
  local output
  output="$("$PRLCTL" -u "$RPC_USER" -P "$RPC_PASS" --simnet --wallet --notls getnewaddress 2>&1)"
  local addr
  addr="$(echo "$output" | grep -oP '(bc1p|tb1p|bcrt1p|sb1p|rprl1p)\w+' | tail -1)"
  [ -n "$addr" ] || die "Could not find a Taproot address in wallet output: $output"
  echo "$addr"
}

# Wait for pearld's RPC to be ready by polling getblockchaininfo
wait_for_pearld() {
  require_file "$PRLCTL" "prlctl"
  local max_attempts=30  # 30 seconds
  local attempt=0
  info "waiting for pearld RPC on 127.0.0.1:18556..."
  while [ $attempt -lt $max_attempts ]; do
    if "$PRLCTL" -u "$RPC_USER" -P "$RPC_PASS" --simnet --notls getblockchaininfo >/dev/null 2>&1; then
      info "pearld RPC is ready"
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  die "pearld RPC did not become ready within ${max_attempts}s"
}

# Wait for wallet's chain client to be connected by polling getnewaddress
# This succeeds only after the wallet's async rpcClientConnectLoop has
# established a connection to pearld and set chainClient via SynchronizeRPC.
wait_for_wallet_chain() {
  require_file "$PRLCTL" "prlctl"
  local max_attempts=30  # 30 seconds
  local attempt=0
  info "waiting for wallet chain client to connect to pearld..."
  while [ $attempt -lt $max_attempts ]; do
    local output
    output="$("$PRLCTL" -u "$RPC_USER" -P "$RPC_PASS" --simnet --wallet --notls getnewaddress 2>&1)" || true
    if echo "$output" | grep -qP '(bc1p|tb1p|bcrt1p|sb1p|rprl1p)\w+'; then
      info "wallet chain client is ready"
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  die "wallet chain client did not connect within ${max_attempts}s"
}

case "$STAGE" in
  create-wallet)
    require_file "$OYSTER" "oyster"
    info "creating wallet in simnet context"
    "$OYSTER" -u "$RPC_USER" -P "$RPC_PASS" --createtemp --simnet --appdata="$WALLET_DATA_DIR"
    ;;

  wallet)
    require_file "$OYSTER" "oyster"
    info "starting oyster wallet on simnet (auto-creates temp wallet on first run)"
    launch_bg "wallet" "$OYSTER" -u "$RPC_USER" -P "$RPC_PASS" --createtemp --simnet --appdata="$WALLET_DATA_DIR" --noservertls --noclienttls
    ;;

  address)
    require_file "$PRLCTL" "prlctl"
    info "requesting a new Taproot mining address from the simnet wallet"
    "$PRLCTL" -u "$RPC_USER" -P "$RPC_PASS" --simnet --wallet --notls getnewaddress
    ;;

  node)
    require_mining_address
    require_file "$PEARLD" "pearld"
    info "starting pearld on simnet"
    launch_bg "node" "$PEARLD" \
      --simnet \
      --notls \
      --rpcuser="$RPC_USER" \
      --rpcpass="$RPC_PASS" \
      --rpclisten=127.0.0.1:18556 \
      --miningaddr="$MINING_ADDRESS" \
      --txindex \
      --debuglevel=debug
    ;;

  gateway)
    require_mining_address
    export PEARLD_RPC_URL="http://127.0.0.1:18556"
    export PEARLD_RPC_USER="$RPC_USER"
    export PEARLD_RPC_PASSWORD="$RPC_PASS"
    export PEARLD_MINING_ADDRESS="$MINING_ADDRESS"
    export MINER_RPC_TRANSPORT="tcp"
    export MINER_RPC_HOST="127.0.0.1"
    export MINER_RPC_PORT="8337"
    info "starting pearl-gateway on tcp://127.0.0.1:8337"
    launch_bg "gateway" "$PEARL_ROOT/miner/.venv/bin/pearl-gateway" start --debug
    ;;

  all)
    # Step 1: Start pearld first so the wallet has a chain backend to connect to
    info "launching pearld in background (without mining address for now)"
    launch_bg "node" "$PEARLD" \
      --simnet \
      --notls \
      --rpcuser="$RPC_USER" \
      --rpcpass="$RPC_PASS" \
      --rpclisten=127.0.0.1:18556 \
      --txindex \
      --debuglevel=debug
    wait_for_pearld

    # Step 2: Start the wallet (needs pearld's RPC to be available before it
    # can connect its async chain client via rpcClientConnectLoop)
    info "launching wallet in background (auto-creates temp wallet on first run)"
    launch_bg "wallet" "$OYSTER" -u "$RPC_USER" -P "$RPC_PASS" --createtemp --simnet --appdata="$WALLET_DATA_DIR" --noservertls --noclienttls

    # Step 3: Wait for the wallet's async chain client to fully connect to
    # pearld (getnewaddress calls requireChainClient() which checks chainClient)
    if [ -z "$MINING_ADDRESS" ]; then
      wait_for_wallet_chain
      info "fetching a Taproot mining address from the running wallet"
      MINING_ADDRESS="$(get_mining_address_from_wallet)"
    fi

    info "mining address: $MINING_ADDRESS"

    # Step 4: Restart pearld with the mining address so it can generate blocks
    info "restarting pearld with mining address..."
    kill "$node_PID" 2>/dev/null
    # Wait for old node log to be closed before reusing the file
    sleep 1
    launch_bg "node" "$PEARLD" \
      --simnet \
      --notls \
      --rpcuser="$RPC_USER" \
      --rpcpass="$RPC_PASS" \
      --rpclisten=127.0.0.1:18556 \
      --miningaddr="$MINING_ADDRESS" \
      --txindex \
      --debuglevel=debug
    wait_for_pearld
    info "pearld restarted with mining address"

    info "to start mining, generate blocks with: prlctl -u $RPC_USER -P $RPC_PASS --simnet --notls generate 101"

    info "launching pearl-gateway in background"
    export PEARLD_RPC_URL="http://127.0.0.1:18556"
    export PEARLD_RPC_USER="$RPC_USER"
    export PEARLD_RPC_PASSWORD="$RPC_PASS"
    export PEARLD_MINING_ADDRESS="$MINING_ADDRESS"
    export MINER_RPC_TRANSPORT="tcp"
    export MINER_RPC_HOST="127.0.0.1"
    export MINER_RPC_PORT="8337"
    launch_bg "gateway" "$PEARL_ROOT/miner/.venv/bin/pearl-gateway" start --debug

    info "simnet stack launched"
    info "wallet:  https://localhost:18554"
    info "pearld:  http://127.0.0.1:18556"
    info "gateway: tcp://127.0.0.1:8337"
    info ""
    info "Logs: /tmp/pearl-wallet.log, /tmp/pearl-node.log, /tmp/pearl-gateway.log"
    info "Stop all: kill $node_PID $wallet_PID $gateway_PID 2>/dev/null; wait"
    ;;

  *)
    echo "Usage: $0 {create-wallet|wallet|address|node|gateway|all} [mining-address]"
    echo ""
    echo "Environment:"
    echo "  PEARL_ROOT            (default: \$HOME/pearl)"
    echo "  RPC_USER              (default: rpcuser)"
    echo "  RPC_PASS              (default: rpcpass)"
    echo "  WALLET_STARTUP_SECONDS (default: 4)"
    exit 1
    ;;
esac
