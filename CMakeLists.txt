cmake_minimum_required(VERSION 3.14)
project(EnerBe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include FetchContent
include(FetchContent)

set(PMT_BUILD_RAPL 1 CACHE BOOL "Enable RAPL" FORCE)
# set(PMT_BUILD_NVML 1 CACHE BOOL "Enable NVML" FORCE)
# Fetch external dependency
FetchContent_Declare(
    pmt
    GIT_REPOSITORY https://git.astron.nl/RD/pmt.git
    GIT_TAG        master  # or a specific commit/tag
)

FetchContent_MakeAvailable(pmt)


# Find OpenMP
find_package(OpenMP REQUIRED)

# Set source and header directories
include_directories(include)

# Collect all source files
set(SOURCES
    src/MM.cpp
    main.cpp
)

# Define the executable
add_executable(EnerBe ${SOURCES})

target_compile_options(EnerBe PRIVATE -Wall -Wextra -O3 -march=native)
target_link_libraries(EnerBe PRIVATE OpenMP::OpenMP_CXX pmt)

