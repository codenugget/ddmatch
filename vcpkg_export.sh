#!/bin/bash
set -x
if [[ ! -d vcpkg ]]; then
	git clone https://github.com/Microsoft/vcpkg.git --depth 1
	git_ret=$?
	if [[ $git_ret -ne 0 ]]; then
		echo "Failed to clone vcpkg!"
	fi
fi

if [[ ! -f vcpkg/vcpkg ]]; then
	pushd vcpkg
	./bootstrap-vcpkg.sh
	boot_ret=$?
	if [[ $boot_ret -ne 0 ]]; then
		echo "Failed to run bootstrap-vcpkg!"
	fi
	popd
fi

pushd vcpkg
	./vcpkg install fftw3:x64-linux gtest:x64-linux
	inst_ret=$?
	if [[ $inst_ret -ne 0 ]]; then
		echo "Failed to install ftw3 and gtest!"
	fi

	./vcpkg export fftw3:x64-linux gtest:x64-linux --raw --output=../export
popd

