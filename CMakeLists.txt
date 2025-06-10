cmake_minimum_required(VERSION 3.10)
project(PlateRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find OpenCV without REQUIRED first
find_package(OpenCV QUIET)

# If not found, try setting OpenCV_DIR manually based on compiler
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found automatically. Attempting manual configuration...")
    set(OpenCV_INCLUDE_DIRS "C:/opencv/build/include")

    if(MSVC)
        set(OpenCV_LIB_DIR "C:/opencv/build/x64/vc16/lib")
        set(OpenCV_LIBS
            opencv_world4110.lib
        )
        file(GLOB OpenCV_DLLS "C:/opencv/build/x64/vc16/bin/*.dll")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(OpenCV_LIB_DIR "C:/opencv/build/x64/mingw/lib")
        file(GLOB OpenCV_LIBS "C:/opencv/build/x64/mingw/lib/libopencv_*.dll.a")
        file(GLOB OpenCV_DLLS "C:/opencv/build/x64/mingw/bin/*.dll")
    else()
        find_package(OpenCV REQUIRED)# Fallback for other compilers
    endif()   
endif()

# Show OpenCV info
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Add SQLite3 as C static library
add_library(sqlite3 STATIC external/sqlite/sqlite3.c)
set_target_properties(sqlite3 PROPERTIES LINKER_LANGUAGE C)
link_directories(${OpenCV_LIB_DIR})

# Main executable and sources
add_executable(PlateRecognition
    src/main.cpp 
    src/database.cpp
    external/inference/inference.cpp
    external/sort/sort.cpp
    external/sort/kalman_box_tracker.cpp
    external/sort/kuhn_munkres.cpp
)

# Include directories
target_include_directories(PlateRecognition PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    incude
    external/sort/include
    external/inference
    external/sql
)

# Link libraries
target_link_libraries(PlateRecognition PRIVATE ${OpenCV_LIBS} sqlite3)

# Copy OpenCV DLLs after build
foreach(dll ${OpenCV_DLLS})
    add_custom_command(TARGET PlateRecognition POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dll}" $<TARGET_FILE_DIR:PlateRecognition>)
endforeach()









