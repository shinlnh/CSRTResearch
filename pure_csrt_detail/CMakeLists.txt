cmake_minimum_required(VERSION 3.15)
project(CSRTFromScratch VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# vcpkg integration (if using vcpkg)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Option to use CUDA
option(USE_CUDA "Enable CUDA support" ON)

# Set OpenCV directory if provided
if(NOT OpenCV_DIR AND EXISTS "C:/opencv/build")
    set(OpenCV_DIR "C:/opencv/build" CACHE PATH "OpenCV directory")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio tracking)

# Check if OpenCV has CUDA support
if(OpenCV_CUDA_VERSION OR DEFINED OpenCV_CUDA)
    message(STATUS "OpenCV with CUDA support found!")
    add_definitions(-DHAVE_OPENCV_CUDA)
    set(HAVE_CUDA ON)
else()
    message(STATUS "OpenCV without CUDA support")
    set(HAVE_CUDA OFF)
endif()

# Find CUDA if available and requested
if(USE_CUDA AND HAVE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA ${CUDA_VERSION} found at ${CUDA_TOOLKIT_ROOT_DIR}")
        add_definitions(-DUSE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
endif()

# Find threading libraries for parallel processing
find_package(Threads REQUIRED)
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "Intel TBB found - enabling parallel processing")
    add_definitions(-DUSE_TBB)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCES
    src/csrt_color_names.cpp
    src/csrt_utils.cpp
    src/csrt_segmentation.cpp
    src/csrt_scale.cpp
    src/csrt_tracker.cpp
)

set(HEADERS
    inc/csrt_color_names.hpp
    inc/csrt_utils.hpp
    inc/csrt_segmentation.hpp
    inc/csrt_scale.hpp
    inc/csrt_tracker.hpp
)

add_library(csrt_lib STATIC ${SOURCES} ${HEADERS})

target_link_libraries(csrt_lib PUBLIC ${OpenCV_LIBS} Threads::Threads)

if(TBB_FOUND)
    target_link_libraries(csrt_lib PUBLIC TBB::tbb)
endif()

if(MSVC)
    target_compile_options(csrt_lib PRIVATE /W4 /O2 /openmp)
else()
    target_compile_options(csrt_lib PRIVATE -Wall -Wextra -O3 -fopenmp)
    target_link_libraries(csrt_lib PUBLIC gomp)
endif()

add_executable(otb_eval src/otb_eval.cpp)
target_link_libraries(otb_eval csrt_lib ${OpenCV_LIBS} Threads::Threads)
if(TBB_FOUND)
    target_link_libraries(otb_eval TBB::tbb)
endif()

add_executable(test_basketball src/test_basketball.cpp)
target_link_libraries(test_basketball csrt_lib ${OpenCV_LIBS} Threads::Threads)
if(TBB_FOUND)
    target_link_libraries(test_basketball TBB::tbb)
endif()

add_executable(otb_compare src/otb_compare.cpp)
target_link_libraries(otb_compare csrt_lib ${OpenCV_LIBS} Threads::Threads)
if(TBB_FOUND)
    target_link_libraries(otb_compare TBB::tbb)
endif()

add_executable(csrt_demo src/main.cpp)
target_link_libraries(csrt_demo csrt_lib ${OpenCV_LIBS} Threads::Threads)
if(TBB_FOUND)
    target_link_libraries(csrt_demo TBB::tbb)
endif()

install(TARGETS csrt_lib csrt_demo
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY inc/ DESTINATION include/csrt)
