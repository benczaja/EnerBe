include_directories(${PROJECT_SOURCE_DIR}/src/)
include(ExternalProject)


find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


# Need this Library to help things out
add_library(HELPER_LIB INTERFACE)

## Here is where you want to add the sources....
set(applications "xgemm;xaxpy")

if (ENABLE_CUDA)
    enable_language(C CXX CUDA)
    set(CMAKE_CUDA_FLAGS "-gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings")
    set(source_suffixs ".c;.cu")
else()
    set(source_suffixs ".c")
endif()

foreach(application IN LISTS applications)
    foreach(source_suffix IN LISTS source_suffixs)

        if(source_suffix MATCHES ".cu")
            set(executable_suffix "_cuda")
        else()
            set(executable_suffix "")
        endif()

        # Add the corresponding suffix per language
        string(CONCAT source "${application}" "${source_suffix}")

        #Double Precision
        string(REGEX REPLACE "^x" "d" executable "${application}")
        string(APPEND executable "${executable_suffix}")

        add_executable("${executable}" "${source}")
        target_link_options("${executable}" PUBLIC "-fopenmp")
        target_link_libraries("${executable}" HELPER_LIB)
        target_compile_definitions("${executable}" PUBLIC USE_DOUBLE)

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        #Single Precision
        string(REGEX REPLACE "^x" "s" executable "${application}")
        string(APPEND executable "${executable_suffix}")
        add_executable("${executable}" "${source}")
        target_link_options("${executable}" PUBLIC "-fopenmp")
        target_link_libraries("${executable}" HELPER_LIB)

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        endforeach()
    endforeach()


if (ENABLE_HIP)

   enable_language(HIP)

    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

    find_package(hip REQUIRED)
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    set(CMAKE_HIP_FLAGS "--offload-arch=gfx90a")

    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

    #DOUBLE PRECISION
    add_executable(dgemm_gpu xgemm.hip)
    # Link Libraries - HIP Device and OpenMP.
    target_link_options(dgemm_gpu PUBLIC "-fopenmp")
    target_link_libraries(dgemm_gpu HELPER_LIB)
    target_compile_definitions(dgemm_gpu PUBLIC USE_DOUBLE)    

    #SINGLE PRECISION
    add_executable(sgemm_gpu xgemm.hip)
    # Link Libraries - HIP Device and OpenMP.
    target_link_options(sgemm_gpu PUBLIC "-fopenmp")
    target_link_libraries(sgemm_gpu HELPER_LIB)


    install(TARGETS dgemm_gpu sgemm_gpu 
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )

endif()



if (ENABLE_PMT)
    message(STATUS "Looking for pmt")
    find_library(PMT_LIBRARY NAMES pmt)
    if (PMT_LIBRARY)
        message(STATUS "Looking for pmt - found")
    else()
        message(STATUS "Looking for pmt - not found\n Configuring PMT to be built locally ")

        set(PMT_CONFIG_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt")
        if (ENABLE_CUDA)
            set(PMT_CONFIG_ARGS "${PMT_CONFIG_ARGS};-DBUILD_NVML_PMT=1")
        endif()
        if (ENABLE_HIP)
            set(PMT_CONFIG_ARGS "${PMT_CONFIG_ARGS};-DBUILD_ROCM_PMT=1")
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

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fpermissive")
    # compile the double precision binaries
    #######################################
    add_executable(dgemm_pmt xgemm_pmt.cpp)
    target_link_libraries(dgemm_pmt HELPER_LIB)
    target_link_libraries(dgemm_pmt pmt)
    target_compile_definitions(dgemm_pmt PUBLIC USE_DOUBLE)

    add_executable(daxpy_pmt axpy_pmt.cpp)
    target_link_libraries(daxpy_pmt HELPER_LIB)
    target_link_libraries(daxpy_pmt pmt)
    target_compile_definitions(daxpy_pmt PUBLIC USE_DOUBLE)    

    # compile the single precision binaries
    #######################################
    add_executable(sgemm_pmt xgemm_pmt.cpp)
    target_link_libraries(sgemm_pmt HELPER_LIB)
    target_link_libraries(sgemm_pmt pmt)


    add_executable(saxpy_pmt axpy_pmt.cpp)
    target_link_libraries(saxpy_pmt HELPER_LIB)
    target_link_libraries(saxpy_pmt pmt)

    install(TARGETS dgemm_pmt sgemm_pmt daxpy_pmt saxpy_pmt
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )

    if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "-gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings")

    #DOUBLE PRECISION
    add_executable(dgemm_pmt_gpu xgemm_pmt.cu)
    target_link_options(dgemm_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(dgemm_pmt_gpu HELPER_LIB)
    target_link_libraries(dgemm_pmt_gpu pmt)
    target_compile_definitions(dgemm_pmt_gpu PUBLIC USE_DOUBLE)

    add_executable(daxpy_pmt_gpu axpy_pmt.cu)
    target_link_options(daxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(daxpy_pmt_gpu HELPER_LIB)
    target_link_libraries(daxpy_pmt_gpu pmt)
    target_compile_definitions(daxpy_pmt_gpu PUBLIC USE_DOUBLE)

    #SINGLE PRECISION
    add_executable(sgemm_pmt_gpu xgemm_pmt.cu)
    target_link_options(sgemm_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(sgemm_pmt_gpu HELPER_LIB)
    target_link_libraries(sgemm_pmt_gpu pmt)

    add_executable(saxpy_pmt_gpu axpy_pmt.cu)
    target_link_options(saxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(saxpy_pmt_gpu HELPER_LIB)
    target_link_libraries(saxpy_pmt_gpu pmt)
    
    install(TARGETS dgemm_pmt_gpu sgemm_pmt_gpu daxpy_pmt_gpu saxpy_pmt_gpu
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
    endif()

    if (ENABLE_HIP)
    enable_language(HIP)

    #DOUBLE PRECISION
    add_executable(dgemm_pmt_gpu xgemm_pmt.hip)
    target_link_options(dgemm_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(dgemm_pmt_gpu HELPER_LIB)
    target_link_libraries(dgemm_pmt_gpu pmt)
    target_compile_definitions(dgemm_pmt_gpu PUBLIC USE_DOUBLE)

    add_executable(daxpy_pmt_gpu axpy_pmt.hip)
    target_link_options(daxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(daxpy_pmt_gpu HELPER_LIB)
    target_link_libraries(daxpy_pmt_gpu pmt)
    target_compile_definitions(daxpy_pmt_gpu PUBLIC USE_DOUBLE)

    #SINGLE PRECISION
    add_executable(sgemm_pmt_gpu xgemm_pmt.hip)
    target_link_options(sgemm_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(sgemm_pmt_gpu HELPER_LIB)
    target_link_libraries(sgemm_pmt_gpu pmt)

    add_executable(saxpy_pmt_gpu axpy_pmt.hip)
    target_link_options(saxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(saxpy_pmt_gpu HELPER_LIB)
    target_link_libraries(saxpy_pmt_gpu pmt)
    
    install(TARGETS dgemm_pmt_gpu sgemm_pmt_gpu daxpy_pmt_gpu saxpy_pmt_gpu
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
    endif()

endif()

