project(cufft LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 35)
endif()

set(src
    add_loop_gpu.cu
)

add_executable(${PROJECT_NAME} add_loop_gpu.cu)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    utils
)

target_link_libraries(${PROJECT_NAME}
    utils
)
