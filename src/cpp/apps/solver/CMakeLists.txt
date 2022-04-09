project(solver)

set(src
    main.cpp
    stb_impl.cpp
)

add_executable(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    ddmatchLib
    utils
)

target_link_libraries(${PROJECT_NAME}
    ddmatchLib
    utils
    fftw3
)
