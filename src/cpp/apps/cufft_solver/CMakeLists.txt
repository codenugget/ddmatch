project(cufft_solver)

set(src
    main.cpp
)

add_executable(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    utils
    cufft
)

target_link_libraries(${PROJECT_NAME}
    utils
    cufft
)
