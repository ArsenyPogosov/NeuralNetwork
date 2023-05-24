cmake_minimum_required(VERSION 3.0)

project(NeuralNetwork VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEURAL_NETWORK_BUILD_TESTS "Build tests for NeuralNetwork library" OFF)

# libraries
include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)
FetchContent_MakeAvailable(Eigen)

# neural
add_library(
  neural
  src/activation/linear.cpp src/activation/relu.cpp src/activation/sigmoid.cpp
  src/activation/soft_max.cpp src/loss/mse.cpp src/loss/cross_entropy.cpp)
target_include_directories(
  neural
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/neural"
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(neural PUBLIC Eigen3::Eigen)

# tests
if(NEURAL_NETWORK_BUILD_TESTS)
  add_subdirectory(tests)
endif()
