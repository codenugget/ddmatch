cmake_minimum_required(VERSION 3.10)

project(ddmatch)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/runtime)

set(CMAKE_CXX_STANDARD 20)

include_directories(
	"src/cpp"
	"external/include"
)

link_directories(
	"external/lib64"
)

add_subdirectory(src)
