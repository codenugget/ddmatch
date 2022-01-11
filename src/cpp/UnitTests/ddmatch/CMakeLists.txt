project(UnitTesting_ddmatch)

if (MSVC)
find_package(GTest CONFIG REQUIRED)
endif()

set(src
  main.cpp
)

add_executable(${PROJECT_NAME} ${src})

add_dependencies(${PROJECT_NAME}
  ddmatchLib
)

if (UNIX)
target_link_libraries(${PROJECT_NAME}
  ddmatchLib
  gtest
  pthread
)
endif()

if (MSVC)
target_link_libraries(${PROJECT_NAME}
  PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
  ddmatchLib
)
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND ${PROJECT_NAME} --gtest_shuffle
)
