cmake_minimum_required(VERSION 3.16)
project(csrt_otb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video tracking)

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

# Find threading library for multi-tasking
find_package(Threads REQUIRED)
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "Intel TBB found - enabling parallel processing")
    add_definitions(-DUSE_TBB)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

add_executable(csrtpure
    pure_csrt/main_pcsrt.cpp)

target_link_libraries(csrtpure PRIVATE ${OpenCV_LIBS} Threads::Threads)

if(TBB_FOUND)
    target_link_libraries(csrtpure PRIVATE TBB::tbb)
endif()
