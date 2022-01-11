project(UnitTesting_core)

if (MSVC)
find_package(GTest CONFIG REQUIRED)
endif()

set(src
  main.cpp
  test_fft.cpp
  test_TCube.cpp
  test_TGrid.cpp
)

add_executable(${PROJECT_NAME} ${src})

if (UNIX)
target_link_libraries(${PROJECT_NAME}
  gtest
  pthread
)
endif()

if (MSVC)
target_link_libraries(${PROJECT_NAME}
  PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
)
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND ${PROJECT_NAME} --gtest_shuffle
)
