FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.3.0)
FetchContent_MakeAvailable(Catch2)
include(CTest)
include(Catch)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(
  tests
  linear.cpp
  relu.cpp
  sigmoid.cpp
  soft_max.cpp
  mse.cpp
  cross_entropy.cpp
  differentiable.cpp
  layer.cpp)
target_include_directories(
  tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include/neural"
                "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_link_libraries(tests PRIVATE neural Catch2::Catch2WithMain)

catch_discover_tests(tests)
