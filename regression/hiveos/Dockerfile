FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 bash jq bc curl ca-certificates cmake g++ libboost-json-dev \
    && rm -rf /var/lib/apt/lists/*
COPY include/ /source/include/
COPY src/broadcast/ /source/src/broadcast/
COPY src/tnn_hip/common/gpu_device_filter.hpp /source/src/tnn_hip/common/gpu_device_filter.hpp
COPY regression/hiveos/ /source/regression/hiveos/
RUN cmake -S /source/regression/hiveos -B /fixture -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /fixture -j2 \
    && dpkg-query -W > /fixture/dependencies.tsv
RUN mkdir -p /input /out /hive/miners/custom /run/hive /var/log/miner/custom
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "/source/regression/hiveos/run.py"]
