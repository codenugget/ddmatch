#!/bin/bash

set -x

if [ ! -d "include" ]
then
	mkdir include
fi


if [ ! -d "lib64" ]
then
	mkdir lib64
fi

pushd .

if [ ! -d "code" ]
then
	mkdir code
fi
cd code

if [ ! -d "googletest" ]
then
	git clone https://github.com/google/googletest
	rm -rf googletest/.git
fi

cd googletest
if [ ! -d build ]
then
	mkdir build
	cd build
	cmake ..
else
	cd build
fi
make
cd ../../..
cp -rn code/googletest/googletest/include/gtest include
cp -rn code/googletest/googlemock/include/gmock include
cp code/googletest/build/lib/* lib64
popd
