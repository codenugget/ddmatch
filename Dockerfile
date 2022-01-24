FROM ubuntu:20.04
MAINTAINER codenugget
# Needed to be set for the cmake package to be installed properly without interactive input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git -y
RUN apt-get install gcc-10 -y
RUN apt-get install g++-10 -y

# Set gcc/g++ so they default to the installed versions
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1
RUN apt-get install cmake -y

# zip / pkg-config were required for vcpkg to build some dependencies
RUN apt-get install zip -y
RUN apt-get install pkg-config -y

# below are convenience for opening a shell terminal in the docker image
RUN apt-get install vim -y
RUN apt-get install curl -y
# enables tab-completion and some colors
RUN curl -L https://raw.githubusercontent.com/docker/compose/1.29.2/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose


RUN mkdir -p /app
# the following two lines will build in vcpkg and the library dependencies we have... optional
RUN cd /app && git clone https://github.com/Microsoft/vcpkg.git --depth 1
RUN cd /app/vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg install gtest:x64-linux fftw3:x64-linux

# copy local files to be built
COPY scripts/build.sh /app/scripts/build.sh
COPY scripts/container_cmake.sh /app/scripts/container_cmake.sh
COPY src /app/src
COPY CMakeLists.txt /app
RUN cd /app/scripts; ./container_cmake.sh

CMD cd /app/scripts; ./build.sh

