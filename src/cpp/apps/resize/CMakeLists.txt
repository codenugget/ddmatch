project(resize)

set(src
    main.cpp
    stb_impl.cpp
)

add_executable(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
