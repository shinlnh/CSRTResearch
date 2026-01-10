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

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio tracking)

# Check if OpenCV has CUDA support
if(OpenCV_CUDA_VERSION)
    message(STATUS "OpenCV with CUDA ${OpenCV_CUDA_VERSION} found!")
    add_definitions(-DHAVE_OPENCV_CUDA)
else()
    message(STATUS "OpenCV without CUDA support")
endif()

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

target_link_libraries(csrt_lib PUBLIC ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(csrt_lib PRIVATE /W4 /O2)
else()
    target_compile_options(csrt_lib PRIVATE -Wall -Wextra -O3)
endif()

add_executable(csrt_demo src/main.cpp)
target_link_libraries(csrt_demo csrt_lib ${OpenCV_LIBS})

add_executable(otb_eval src/otb_eval.cpp)
target_link_libraries(otb_eval csrt_lib ${OpenCV_LIBS})

add_executable(otb_eval_simple src/otb_eval_simple.cpp)
target_link_libraries(otb_eval_simple csrt_lib ${OpenCV_LIBS})
target_link_libraries(otb_eval_simple Threads::Threads)

add_executable(test_basketball src/test_basketball.cpp)
target_link_libraries(test_basketball csrt_lib ${OpenCV_LIBS})

# Build otb_compare with multi-threading support
add_executable(otb_compare src/otb_compare.cpp)
target_link_libraries(otb_compare csrt_lib ${OpenCV_LIBS})

# Add threading support for parallel processing
find_package(Threads REQUIRED)
target_link_libraries(otb_compare Threads::Threads)

install(TARGETS csrt_lib csrt_demo
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY inc/ DESTINATION include/csrt)
