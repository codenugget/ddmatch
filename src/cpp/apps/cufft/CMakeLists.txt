project(cufft)

set(src
    simpleCall.cu
)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${src})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    utils
)

target_link_libraries(${PROJECT_NAME}
    utils
)
