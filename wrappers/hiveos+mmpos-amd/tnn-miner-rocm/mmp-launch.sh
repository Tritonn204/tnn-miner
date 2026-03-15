#!/usr/bin/env bash

EXEC="./tnn-miner-rocm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Persist HIPRTC compiled kernel cache next to the miner
export TNN_HIP_CACHE_PATH="$SCRIPT_DIR/.hip_cache"
mkdir -p "$TNN_HIP_CACHE_PATH"

ARGS=("$@")
FINAL_ARGS=()

POOL_VAL=""
USER_VAL=""
PASS_VAL=""
ALGO_VAL=""

# ---------------- PARSE ARGS ----------------
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --pool)     POOL_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --user)     USER_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --password) PASS_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --algo)     ALGO_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        *)          FINAL_ARGS+=("${ARGS[$i]}"); ((i+=1)) ;;
    esac
done

# ---------------- DEFAULTS ----------------
[[ -z "$ALGO_VAL" ]] && ALGO_VAL="--XEL"
[[ -z "$PASS_VAL" ]] && PASS_VAL="x"

# ---------------- BUILD MINER CMD ----------------
CMD=( "$EXEC" )

CMD+=( "$ALGO_VAL" )

[[ -n "$POOL_VAL" ]] && CMD+=( --daemon-address "$POOL_VAL" )
[[ -n "$USER_VAL" ]] && CMD+=( --wallet "$USER_VAL" )
[[ -n "$PASS_VAL" ]] && CMD+=( --pass "$PASS_VAL" )

CMD+=( --mmpos )

# pass-through extra args
CMD+=( "${FINAL_ARGS[@]}" )

# ---------------- RUN ----------------
exec "${CMD[@]}"
