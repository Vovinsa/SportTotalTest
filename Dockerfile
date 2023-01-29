FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libmicrohttpd-dev \
    cmake \
    build-essential \
    g++

WORKDIR SportTotalTest/

ADD . .

RUN mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j1

CMD ["./build/SportTotalTest"]