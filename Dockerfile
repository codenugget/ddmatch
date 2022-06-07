FROM ubuntu:20.04
# Needed to be set for the cmake package to be installed properly without interactive input
# Set gcc/g++ so they default to the installed versions
# zip / pkg-config were required for vcpkg to build some dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git gcc-10 g++-10 cmake zip pkg-config curl && \
 update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1 && \
 update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1

WORKDIR /app
# the following two lines will build in vcpkg and the library dependencies we have... optional
RUN git clone https://github.com/Microsoft/vcpkg.git --depth 1 && \
cd vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg install gtest:x64-linux fftw3:x64-linux matplotplusplus:x64-linux nlohmann-json:x64-linux cuda:x64-linux

# copy local files to be built
COPY scripts/build.sh /app/scripts/build.sh
COPY scripts/docker_cmake.sh /app/scripts/docker_cmake.sh
COPY src /app/src
COPY CMakeLists.txt /app
RUN cd scripts && ./docker_cmake.sh

CMD cd scripts && ./build.sh

