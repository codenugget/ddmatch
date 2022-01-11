pushd .
cd ..
if not exist build (
	mkdir build
)
cd build
if not exist ddmatch.sln (
	cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
)
popd
