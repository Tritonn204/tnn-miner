#!/usr/bin/env bash

EXEC="./tnn-miner-cuda"

ARGS=("$@")
FINAL_ARGS=()

POOL_VAL=""
POOL_HOST=""
POOL_PORT=""

USER_VAL=""
PASS_VAL=""
ALGO_VAL=""

# ---------------- PARSE ARGS ----------------
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --pool)
            POOL_VAL="${ARGS[$((i+1))]}"
            ((i+=2))
            ;;
        --user)
            USER_VAL="${ARGS[$((i+1))]}"
            ((i+=2))
            ;;
        --password)
            PASS_VAL="${ARGS[$((i+1))]}"
            ((i+=2))
            ;;
        --algo)
            ALGO_VAL="${ARGS[$((i+1))]}"
            ((i+=2))
            ;;
        *)
            FINAL_ARGS+=("${ARGS[$i]}")
            ((i+=1))
            ;;
    esac
done

# ---------------- DEFAULTS ----------------
[[ -z "$ALGO_VAL" ]] && ALGO_VAL="--XEL"
[[ -z "$PASS_VAL" ]] && PASS_VAL="x"

# ---------------- PARSE pool:port ----------------
if [[ "$POOL_VAL" == *:* ]]; then
    POOL_HOST="${POOL_VAL%%:*}"
    POOL_PORT="${POOL_VAL##*:}"
else
    POOL_HOST="$POOL_VAL"
fi

# ---------------- BUILD MINER CMD ----------------
CMD=( "$EXEC" )

CMD+=( "$ALGO_VAL" )

[[ -n "$POOL_HOST" ]] && CMD+=( --stratum --daemon-address "$POOL_HOST" )
[[ -n "$POOL_PORT" ]] && CMD+=( --port "$POOL_PORT" )

[[ -n "$USER_VAL" ]] && CMD+=( --wallet "$USER_VAL" )
[[ -n "$PASS_VAL" ]] && CMD+=( --pass "$PASS_VAL" )

CMD+=( --mmpos )

# pass-through extra args
CMD+=( "${FINAL_ARGS[@]}" )

# ---------------- RUN ----------------
exec "${CMD[@]}"
