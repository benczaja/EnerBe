include_directories(${PROJECT_SOURCE_DIR}/src/cuda)

## Here is where you want to add the sources....
set(applications "xgemm;xaxpy")

if (ENABLE_PMT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fpermissive")
    list(TRANSFORM applications APPEND "_pmt")
endif()

enable_language(C CXX CUDA)
set(CMAKE_CUDA_FLAGS "-Xcompiler=-fopenmp -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings")
set(source_suffixs ".cu")





foreach(application IN LISTS applications)
    foreach(source_suffix IN LISTS source_suffixs)
        
        set(executable_suffix "_cuda")

        # Add the corresponding suffix per language
        string(CONCAT source "${application}" "${source_suffix}")

        #Double Precision
        string(REGEX REPLACE "^x" "d" executable "${application}")
        string(APPEND executable "${executable_suffix}")

        add_executable("${executable}" "${source}")
        target_link_options("${executable}" PUBLIC "-fopenmp")
        target_link_libraries("${executable}" HELPER_LIB)
        if (ENABLE_PMT)
            target_link_libraries("${executable}" pmt)
        endif()
        target_compile_definitions("${executable}" PUBLIC USE_DOUBLE)

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        #Single Precision
        string(REGEX REPLACE "^x" "s" executable "${application}")
        string(APPEND executable "${executable_suffix}")
        add_executable("${executable}" "${source}")
        target_link_options("${executable}" PUBLIC "-fopenmp")
        target_link_libraries("${executable}" HELPER_LIB)

        if (ENABLE_PMT)
            target_link_libraries("${executable}" pmt)
        endif()

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        endforeach()
    endforeach()

