include(ExternalProject)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

file(GLOB SOURCES "*.cpp")



if (ENABLE_PMT)
    message(STATUS "Looking for pmt")
    find_library(PMT_LIBRARY NAMES pmt)
    if (PMT_LIBRARY)
        message(STATUS "Looking for pmt - found")
    else()
        message(STATUS "Looking for pmt - not found\n Configuring PMT to be built locally ")

        if (ENABLE_CUDA)
            set(PMT_CONFIG_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt;-DPMT_BUILD_RAPL=1;-DPMT_BUILD_NVML=1;-DPMT_BUILD_NVIDIA=1")
        elseif(ENABLE_HIP)
            set(PMT_CONFIG_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt;-DPMT_BUILD_RAPL=1;-DPMT_BUILD_ROCM=1")
        else()
            set(PMT_CONFIG_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt;-DPMT_BUILD_RAPL=1;-DPMT_BUILD_NVML=0;-DPMT_BUILD_ROCM=0;-DPMT_BUILD_NVIDIA=0")
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

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "-Xcompiler=-fopenmp -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings")
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
endif()



add_executable(dEnerBe ${SOURCES})
target_compile_definitions(dEnerBe PUBLIC USE_DOUBLE)
include_directories("${PROJECT_SOURCE_DIR}/OpenBLAS/include")
#target_link_libraries(dEnerBe lapack)
#target_link_libraries(dEnerBe openblas)


if (ENABLE_PMT)
target_link_libraries(dEnerBe pmt)
target_compile_definitions(dEnerBe PUBLIC PMT_ENABLED)
endif()

if (ENABLE_CUDA)
target_link_options(dEnerBe PUBLIC "-fopenmp")
target_compile_definitions(dEnerBe PUBLIC CUDA_ENABLED)
endif()

if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/home/benjamic/EnerBe/OpenBLAS/include/)
    target_link_libraries(dEnerBe ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

install(TARGETS dEnerBe RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)


add_executable(sEnerBe ${SOURCES})
target_compile_definitions(sEnerBe PUBLIC)

if (ENABLE_PMT)
    target_link_libraries(sEnerBe pmt)
    target_compile_definitions(sEnerBe PUBLIC PMT_ENABLED)
endif()

if (ENABLE_CUDA)
    target_link_options(sEnerBe PUBLIC "-fopenmp")
    target_compile_definitions(sEnerBe PUBLIC CUDA_ENABLED)
endif()

if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/home/benjamic/EnerBe/OpenBLAS/include/)
    target_link_libraries(sEnerBe ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

install(TARGETS sEnerBe RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

