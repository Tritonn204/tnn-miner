#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "${SCRIPT_DIR}/" &> /dev/null && pwd )


docker build --progress=plain -f ${ROOT_DIR}/docker/Dockerfile.windows . -o ${ROOT_DIR}/export
exit $?
