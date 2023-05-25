cmake_minimum_required(VERSION 3.0)

project(NeuralNetwork VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEURAL_NETWORK_BUILD_TESTS "Build tests for NeuralNetwork library" OFF)

set(NEURAL_NETWORK_DEFAULT_BATCH_SIZE "10" CACHE STRING "")
set(NEURAL_NETWORK_DEFAULT_STARTING_LEARNING_RATE
	"0.01" CACHE STRING "")
set(NEURAL_NETWORK_DEFAULT_LEARNING_RATE_DECAY "0.01" CACHE STRING "")

configure_file(src/defaults.h.in defaults.h)

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
  src/activation/linear.cpp
  src/activation/relu.cpp
  src/activation/sigmoid.cpp
  src/activation/soft_max.cpp
  src/loss/mse.cpp
  src/loss/cross_entropy.cpp
  src/layer.cpp
  src/neural_network_impl.cpp)
target_include_directories(
  neural
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include/neural"
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(neural PUBLIC Eigen3::Eigen)

# tests
if(NEURAL_NETWORK_BUILD_TESTS)
  add_subdirectory(tests)
endif()
