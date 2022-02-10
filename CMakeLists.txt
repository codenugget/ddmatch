cmake_minimum_required(VERSION 3.10)

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg)
	message( FATAL_ERROR
		"You must clone vcpkg at ${CMAKE_CURRENT_SOURCE_DIR}\n"
		"cd ${CMAKE_CURRENT_SOURCE_DIR}\n"
		"git clone https://github.com/Microsoft/vcpkg.git --depth 1\n"
		"cd vcpkg\n"
		"./bootstrap-vcpkg.sh"
		"./vcpkg install gtest:x64-linux fftw3:x64-linux"
		)
endif()

if (WIN32)
	set(CMAKE_GENERATOR_PLATFORM x64) # Force 64 bits
elseif (LINUX)
	set(VCPKG_TARGET_TRIPLET "x64-linux")
endif()

# Setup vcpkg script with CMake (note: before project())
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}) # Force install directory
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/runtime)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

project(ddmatch)

find_package(GTest CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)

include_directories(
	"src/cpp"
)

add_subdirectory(src)
