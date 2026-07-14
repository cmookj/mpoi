cmake_minimum_required(VERSION 3.25)
project(mpoi VERSION 0.0.1
             DESCRIPTION "A C++ multi-processing object interface library"
             LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(Dependencies.cmake)

# Main library target
# ------------------------------------------------------------------------------
add_library(mpoi STATIC
    ${PROJECT_SOURCE_DIR}/src/mpoi.cpp)

target_include_directories(mpoi PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

find_library(OPENCL_FRAMEWORK OpenCL)

target_link_libraries(mpoi PRIVATE ${OPENCL_FRAMEWORK})

# Test target
# ------------------------------------------------------------------------------
# add_subdirectory(tests)

# Examples
# ------------------------------------------------------------------------------
add_subdirectory(examples)
