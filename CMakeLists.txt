cmake_minimum_required(VERSION 3.14)
project(EnerBe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA if available
if(ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled.")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

####################################
# All of this is for PMT
####################################
# Include FetchContent
include(FetchContent)
#Largerly stealing this from PMT's CMakeLists.txt
option(PMT_BUILD_CRAY "" OFF)
option(PMT_BUILD_LIKWID "" OFF)
option(PMT_BUILD_NVML "" OFF)
option(PMT_BUILD_POWERSENSOR2 "" OFF)
option(PMT_BUILD_POWERSENSOR3 "" OFF)
option(PMT_BUILD_PYTHON "" OFF)
option(PMT_BUILD_RAPL "" ON)
option(PMT_BUILD_ROCM "" OFF)
option(PMT_BUILD_AMDSMI "" OFF)
option(PMT_BUILD_TEGRA "" OFF)
option(PMT_BUILD_XILINX "" OFF)
option(PMT_BUILD_NVIDIA "" OFF)
option(PMT_BUILD_BINARY "" OFF)

# Include FetchContent
include(FetchContent)

set(PMT_BUILD_RAPL 1 CACHE BOOL "Enable RAPL" FORCE)
if(ENABLE_CUDA)
    set(PMT_BUILD_NVML 1 CACHE BOOL "Enable CUDA" FORCE)
endif()
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

# Optionally add CUDA source
if(ENABLE_CUDA)
    list(APPEND SOURCES src/MM.cu)
endif()

# Define the executable
add_executable(EnerBe ${SOURCES})

target_compile_options(EnerBe PRIVATE -Wall -Wextra -O3 -march=native)
target_link_libraries(EnerBe PRIVATE OpenMP::OpenMP_CXX pmt)
set_target_properties(EnerBe PROPERTIES INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib"
)
if(ENABLE_CUDA)
    target_compile_definitions(EnerBe PRIVATE ENABLE_CUDA=1)
endif()


install(TARGETS EnerBe pmt
    ARCHIVE DESTINATION "${CMAKE_SOURCE_DIR}/lib"        # For static libraries
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/lib"        # For shared libraries
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/bin"        # For executables
    PUBLIC_HEADER DESTINATION include
)

