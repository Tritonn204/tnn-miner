#!/usr/bin/env bash
set -euo pipefail
# No host directories or Docker socket are mounted into the test container.
archive=$(realpath "$1")
linux_archive=$(realpath "$2")
platform=${4:-hiveos}
case "$platform" in hiveos|mmpos) ;; *) echo 'Unknown platform' >&2; exit 2 ;; esac
output=$(realpath -m "${3:-build/$platform-results}")
image="tnn-$platform-test:${CI_JOB_ID:-local}"
container="tnn-$platform-test-${CI_JOB_ID:-$$}"
mkdir -p "$output"
# shellcheck disable=SC2317 # Invoked by the EXIT trap, including build failures.
cleanup() {
  docker cp "$container:/out/." "$output/" >/dev/null 2>&1 || true
  if [[ ! -f "$output/junit.xml" ]]; then
    printf '%s\n' '<testsuite name="hiveos-integration" tests="1" failures="1"><testcase name="harness-startup"><failure>Harness did not produce results; inspect CI build/container logs.</failure></testcase></testsuite>' > "$output/junit.xml"
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker image rm "$image" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker build -f regression/hiveos/Dockerfile -t "$image" .
docker image inspect "$image" > "$output/container-image.json"
docker create --name "$container" --network none --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --cpus 2 --memory 2g \
  -e HIVEOS_TEST_ISOLATED=1 "$image" \
  --platform "$platform" \
  --archive "/input/$(basename "$archive")" \
  --linux-archive "/input/$(basename "$linux_archive")"
docker cp "$archive" "$container:/input/"
docker cp "$linux_archive" "$container:/input/"
docker start -a "$container"
code=$(docker inspect -f '{{.State.ExitCode}}' "$container")
exit "$code"
