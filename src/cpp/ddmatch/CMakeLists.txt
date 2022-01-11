project (ddmatchLib)

set(CMAKE_DEBUG_POSTFIX d)

set(src
  DiffeoFunctionMatching.cpp
  DiffeoFunctionMatching.h
  Diffeo_functions.cpp
  Diffeo_functions.h
)
set(headers )

add_library(${PROJECT_NAME} ${src})

#add_dependencies(${PROJECT_NAME}
#    MathLib
#)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${headers}")
INSTALL(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_SOURCE_DIR}/stash/lib
    PUBLIC_HEADER DESTINATION  ${CMAKE_SOURCE_DIR}/stash/include/ImageLib
)
