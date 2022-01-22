project (ddmatchLib)
set(CMAKE_DEBUG_POSTFIX d)


set(src
  DiffeoFunctionMatching.cpp
  DiffeoFunctionMatching.h
  Diffeo_functions.cpp
  Diffeo_functions.h
)

add_library(${PROJECT_NAME} ${src})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
