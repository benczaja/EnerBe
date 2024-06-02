include_directories(${PROJECT_SOURCE_DIR}/src/cpp)


## Here is where you want to add the sources....
set(applications "xEnerBe")

if (ENABLE_PMT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fpermissive")
    list(TRANSFORM applications APPEND "_pmt")
endif()

set(source_suffixs ".cpp")




foreach(application IN LISTS applications)
    foreach(source_suffix IN LISTS source_suffixs)

        # Add the corresponding suffix per language
        string(CONCAT source "${application}" "${source_suffix}")

        #Double Precision (Prepend string with d)
        string(REGEX REPLACE "^x" "d" executable "${application}")

        add_executable("${executable}" "${source}")
        target_link_libraries("${executable}" HELPER_LIB)
        if (ENABLE_PMT)
            target_link_libraries("${executable}" pmt)
        endif()
        target_compile_definitions("${executable}" PUBLIC USE_DOUBLE)

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        #Single Precision
        string(REGEX REPLACE "^x" "s" executable "${application}")
        add_executable("${executable}" "${source}")
        target_link_libraries("${executable}" HELPER_LIB)
        if (ENABLE_PMT)
            target_link_libraries("${executable}" pmt)
        endif()

        install(TARGETS "${executable}" RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

        endforeach()
    endforeach()

