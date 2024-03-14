cmake_minimum_required(VERSION 3.24)

project(example LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "$ENV{MAX_PKG_DIR}/lib/cmake")
include(AddMaxEngine)

add_executable(basics main.c)
target_link_libraries(basics PUBLIC max-engine)
