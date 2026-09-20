#!/usr/bin/env bash

set -euo pipefail

# mmpOS renames the package directory. Resolve everything beside this script.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
if [[ -d "$ROOT/libs" ]]; then
    export LD_LIBRARY_PATH="$ROOT/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

POOL="" WALLET="" PASSWORD="x" COIN="" ALGO="" PROTOCOL="tcp"
EXTRA=()
EXPLICIT_ALGO=false
is_coin() {
    case "${1,,}" in
        dero|xel|spr|rx0|xmr|sal|zeph|vrsc|aix|nxl|htn|wala|shai|advc|xtm|rin|tdc|ytn|gold|mtbc|mgpc|urx|crnc|ysc|eqpay|lpepe|kawpow|rvn|quai|qubit|prl)
            return 0 ;;
        *) return 1 ;;
    esac
}
need_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
    if [[ "$1" != --algo && "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --pool|--user|--password|--coin|--algo|--api-port)
            need_value "$@"
            case "$1" in
                --pool) POOL=$2 ;;
                --user) WALLET=$2 ;;
                --password) PASSWORD=$2 ;;
                --coin) COIN=${2^^} ;;
                --algo) ALGO=$2 ;;
                --api-port)
                    if [[ "$2" != 8989 ]]; then
                        echo "TNN uses API port 8989; set the mmpOS profile API port to 8989 (requested $2)." >&2
                    fi ;;
            esac
            shift 2 ;;
        tcp|tls) PROTOCOL=$1; shift ;;
        *)
            # Explicit miner algorithm flags override the profile's default.
            if [[ "$1" == --* ]] && is_coin "${1#--}"; then
                EXPLICIT_ALGO=true
            fi
            case "${1,,}" in
                --xel-v[123]|--xel=v[123]|--randomx|--yespower|--yespower=*)
                    EXPLICIT_ALGO=true ;;
            esac
            EXTRA+=("$1"); shift
            # Preserve an option's value even if it happens to be 'tcp'/'tls'.
            if (($#)) && [[ "${EXTRA[-1]}" == -* && "$1" != -* ]]; then
                EXTRA+=("$1"); shift
            fi ;;
    esac
done

CMD=("$ROOT/tnn-miner")
if ! "$EXPLICIT_ALGO"; then
    if [[ -n "$ALGO" ]]; then
        CMD+=("--${ALGO#--}")
    else
        if is_coin "$COIN"; then
            CMD+=("--${COIN,,}")
        else
            CMD+=(--XEL)
        fi
    fi
fi

if [[ -n "$POOL" ]]; then
    if [[ "$POOL" != *://* ]]; then
        SCHEME=stratum+tcp
        [[ "$PROTOCOL" == tls ]] && SCHEME=stratum+ssl
        POOL="$SCHEME://$POOL"
    fi
    CMD+=(--daemon-address "$POOL")
fi
[[ -n "$WALLET" ]] && CMD+=(--wallet "$WALLET")
CMD+=(--password "$PASSWORD" --mmpos)
# The miner supplies its thread default; never duplicate an explicit --threads.
exec "${CMD[@]}" "${EXTRA[@]}"
