project(UnitTesting_core)

find_package(GTest CONFIG REQUIRED)

set(src
  main.cpp
  test_fft.cpp
  test_TCube.cpp
  test_TGrid.cpp
)

add_executable(${PROJECT_NAME} ${src})

target_link_libraries(${PROJECT_NAME}
  PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
  pthread
)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND ${PROJECT_NAME} --gtest_shuffle
)
