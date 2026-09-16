# CPU-only build of gchlebus/tennis-court-detection @ 762d077541a77abf4923f5f8f689a1410927d35e
# with the minimal patch in patches/tcd_opencv4.patch (Conan removed, OpenCV 4 constants).
# No CUDA / no GPU: base image is plain Ubuntu, OpenCV comes from Ubuntu's CPU-only package.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config libopencv-dev time ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY probe.cpp /build/probe.cpp
COPY tcd-src /build/tcd-src

RUN pkg-config --modversion opencv4 \
 && cmake -S /build/tcd-src -B /build/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /build/build --parallel \
 && g++ -O2 -std=c++11 -o /usr/local/bin/tcd_probe /build/probe.cpp $(pkg-config --cflags --libs opencv4)

WORKDIR /work
