include_directories(${PROJECT_SOURCE_DIR}/src/hip)

## Here is where you want to add the sources....
set(applications "xgemm;xaxpy")

if (ENABLE_PMT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fpermissive")
    list(TRANSFORM applications APPEND "_pmt")
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

enable_language(C CXX HIP)
find_package(hip REQUIRED)

set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
set(CMAKE_HIP_FLAGS "--offload-arch=gfx90a -fopenmp")

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

set(source_suffixs ".hip")





foreach(application IN LISTS applications)
    foreach(source_suffix IN LISTS source_suffixs)

        set(executable_suffix "_hip")

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

