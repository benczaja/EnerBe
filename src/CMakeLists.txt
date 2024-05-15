include(ExternalProject)

find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


add_library(HELPER_LIB INTERFACE)

if (ENABLE_PMT)
    message(STATUS "Looking for pmt")
    find_library(PMT_LIBRARY NAMES pmt)
    if (PMT_LIBRARY)
        message(STATUS "Looking for pmt - found")
    else()
        message(STATUS "Looking for pmt - not found\n Configuring PMT to be built locally ")

        set(PMT_CONFIG_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt;-DPMT_BUILD_RAPL=1")
        if (ENABLE_CUDA)
            set(PMT_CONFIG_ARGS "${PMT_CONFIG_ARGS};-DPMT_BUILD_RAPL=1;-DPMT_BUILD_NVML=1")
        endif()
        if (ENABLE_HIP)
            set(PMT_CONFIG_ARGS "${PMT_CONFIG_ARGS};-DPMT_BUILD_RAPL=1;-DPMT_BUILD_ROCM=1")
        endif()
        message(STATUS "PMT will be configured with these arguments: ")
        message(STATUS "${PMT_CONFIG_ARGS} ")

        ExternalProject_Add(PMT_LIBRARY
	    INSTALL_DIR			"${PROJECT_SOURCE_DIR}/pmt"
        GIT_REPOSITORY      "https://git.astron.nl/RD/pmt.git" 
        CMAKE_ARGS          "${PMT_CONFIG_ARGS}"
	    )
        include_directories(${PROJECT_SOURCE_DIR}/pmt/include)
        link_directories(${PROJECT_SOURCE_DIR}/pmt/lib)
        link_directories(${PROJECT_SOURCE_DIR}/pmt/lib64)
        
        # just to go a bit Nuclear if we are running into EESSI compat layer issues.
        set(CMAKE_BUILD_RPATH "${PROJECT_SOURCE_DIR}/pmt/lib64" "${PROJECT_SOURCE_DIR}/pmt/lib")
        set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/pmt/lib64" "${PROJECT_SOURCE_DIR}/pmt/lib")

    endif()
endif()


add_subdirectory(cpp)

if (ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

if (ENABLE_HIP)
    add_subdirectory(hip)
endif()
