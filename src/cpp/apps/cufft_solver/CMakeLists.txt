project(cufft_solver)

set(src
    main.cpp
    stb_impl.cpp
)

find_package(argparse CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    utils
    ddmatch_gpu
)

target_link_libraries(${PROJECT_NAME}
    utils
    ddmatch_gpu
    argparse::argparse
)

