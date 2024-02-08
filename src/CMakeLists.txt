include_directories(${PROJECT_SOURCE_DIR}/src/)
include(ExternalProject)



find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

add_library(HELPER_LIB INTERFACE)


# compile the double precision binaries
#######################################
add_executable(dgemm xgemm.c)
target_link_libraries(dgemm HELPER_LIB)
target_compile_definitions(dgemm PUBLIC USE_DOUBLE)


add_executable(daxpy axpy.c)
target_link_libraries(daxpy HELPER_LIB)
target_compile_definitions(daxpy PUBLIC USE_DOUBLE)

# compile the single precision binaries
#######################################
add_executable(sgemm xgemm.c)
target_link_libraries(sgemm HELPER_LIB)

add_executable(saxpy axpy.c)
target_link_libraries(saxpy HELPER_LIB)

install(TARGETS dgemm sgemm daxpy saxpy 
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "-gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings")
    
    #DOUBLE PRECISION
    add_executable(dgemm_gpu xgemm.cu)
    target_link_options(dgemm_gpu PUBLIC "-fopenmp")
    target_link_libraries(dgemm_gpu HELPER_LIB)
    target_compile_definitions(dgemm_gpu PUBLIC USE_DOUBLE)

    add_executable(daxpy_gpu axpy.cu)
    target_link_options(daxpy_gpu PUBLIC "-fopenmp")
    target_link_libraries(daxpy_gpu HELPER_LIB)
    target_compile_definitions(daxpy_gpu PUBLIC USE_DOUBLE)

    #SINGLE PRECISION
    add_executable(sgemm_gpu xgemm.cu)
    target_link_options(sgemm_gpu PUBLIC "-fopenmp")
    target_link_libraries(sgemm_gpu HELPER_LIB)

    add_executable(saxpy_gpu axpy.cu)
    target_link_options(saxpy_gpu PUBLIC "-fopenmp")
    target_link_libraries(saxpy_gpu HELPER_LIB)
    
    install(TARGETS dgemm_gpu sgemm_gpu daxpy_gpu saxpy_gpu
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )

endif()


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
    #set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${PMT_DIR}/lib)
    #include_directories(${PMT_DIR}/include)
    find_library(PMT_LIBRARY NAMES pmt)
    if (PMT_LIBRARY)
        message(STATUS "Looking for pmt - found")
    else()
        #message(FATAL_ERROR "Looking for pmt - not found\n You may need to set -DPMT_DIR to the local installation of PMT ")
        message("Ben")
        ExternalProject_Add(PMT_LIBRARY
	    INSTALL_DIR			"${PROJECT_SOURCE_DIR}/pmt"
        #GIT_TAG             "master"
        GIT_REPOSITORY      "https://git.astron.nl/RD/pmt.git"
	    #CONFIGURE_COMMAND	"git submodule update --init"
        CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/pmt
	    )
        #find_library(PMT_LIBRARY NAMES pmt)
        include_directories(${PROJECT_SOURCE_DIR}/pmt/include)
        #set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${PROJECT_SOURCE_DIR}/pmt/lib64)
        #add_library(pmt STATIC ${PROJECT_SOURCE_DIR}/pmt)
        #set(pmt_ROOT ${PROJECT_SOURCE_DIR}/pmt)
        #set(LINK_FLAGS "-L${PROJECT_SOURCE_DIR}/pmt/lib64")
        set(CMAKE_BUILD_RPATH "${PROJECT_SOURCE_DIR}/pmt/lib64")
        set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/pmt/lib64")

    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fpermissive -L${PROJECT_SOURCE_DIR}/pmt/lib64 ")
    # compile the double precision binaries
    #######################################
    add_executable(dgemm_pmt xgemm_pmt.cpp)
    target_link_libraries(dgemm_pmt HELPER_LIB)
    target_link_libraries(dgemm_pmt pmt)
    target_compile_definitions(dgemm_pmt PUBLIC USE_DOUBLE)
    #set_target_properties(dgemm_pmt PROPERTIES INSTALL_RPATH "${PROJECT_SOURCE_DIR}/pmt/lib64")
    #set_target_properties(dgemm_pmt PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
    #target_link_options(dgemm_pmt PUBLIC "-lpmt")

    add_executable(daxpy_pmt axpy_pmt.cpp)
    target_link_libraries(daxpy_pmt HELPER_LIB)
    target_compile_definitions(daxpy_pmt PUBLIC USE_DOUBLE)    
    target_link_options(daxpy_pmt PUBLIC "-lpmt")

    # compile the single precision binaries
    #######################################
    add_executable(sgemm_pmt xgemm_pmt.cpp)
    target_link_libraries(sgemm_pmt HELPER_LIB)
    target_link_options(sgemm_pmt PUBLIC "-lpmt")

    add_executable(saxpy_pmt axpy_pmt.cpp)
    target_link_libraries(saxpy_pmt HELPER_LIB)
    target_link_options(saxpy_pmt PUBLIC "-lpmt")

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
    target_compile_definitions(dgemm_pmt_gpu PUBLIC USE_DOUBLE)
    target_link_options(dgemm_pmt_gpu PUBLIC "-lpmt")

    add_executable(daxpy_pmt_gpu axpy_pmt.cu)
    target_link_options(daxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(daxpy_pmt_gpu HELPER_LIB)
    target_compile_definitions(daxpy_pmt_gpu PUBLIC USE_DOUBLE)
    target_link_options(daxpy_pmt_gpu PUBLIC "-lpmt")

    #SINGLE PRECISION
    add_executable(sgemm_pmt_gpu xgemm_pmt.cu)
    target_link_options(sgemm_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(sgemm_pmt_gpu HELPER_LIB)
    target_link_options(sgemm_pmt_gpu PUBLIC "-lpmt")

    add_executable(saxpy_pmt_gpu axpy_pmt.cu)
    target_link_options(saxpy_pmt_gpu PUBLIC "-fopenmp")
    target_link_libraries(saxpy_pmt_gpu HELPER_LIB)
    target_link_options(saxpy_pmt_gpu PUBLIC "-lpmt")
    
    install(TARGETS dgemm_pmt_gpu sgemm_pmt_gpu daxpy_pmt_gpu saxpy_pmt_gpu
    RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
    )
    endif()
endif()

