cmake_minimum_required(VERSION 3.14)
project(EnerBe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set source and header directories
include_directories(include)

# Collect all source files
set(SOURCES
    src/MM.cpp
    main.cpp
)

# Define the executable
add_executable(EnerBe ${SOURCES})

target_compile_options(EnerBe PRIVATE -Wall -Wextra -O3)

