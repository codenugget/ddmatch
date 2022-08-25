#!/bin/bash
pushd ../vcpkg && ./vcpkg install gtest:x64-linux fftw3:x64-linux matplotplusplus:x64-linux nlohmann-json:x64-linux argparse:x64-linux
popd
