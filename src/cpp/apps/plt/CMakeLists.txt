project(plt)

set(src
    main.cpp
)

add_executable(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_dependencies(${PROJECT_NAME}
    ddmatchLib
)

target_link_libraries(${PROJECT_NAME}
    ddmatchLib
    fftw3
    Matplot++::cimg std::filesystem Matplot++::matplot Matplot++::nodesoup 
)
