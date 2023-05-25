add_executable(example main.cpp)

target_include_directories(example PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(example PRIVATE neural)

