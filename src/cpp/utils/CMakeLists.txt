project (utils)

set(src
  to_image.cpp
  to_image.h
)

add_library(${PROJECT_NAME} ${src})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
