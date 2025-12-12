cmake_minimum_required(VERSION 3.15)
project(UpdatedCSRT VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force MinGW compiler (compatible with MSYS2 OpenCV)
if(WIN32)
    set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")
    set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV
find_package(OpenCV REQUIRED 
    COMPONENTS core imgproc video highgui dnn objdetect tracking
)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/HOGExtractor.cpp
    src/DeepFeatureExtractor.cpp
    src/CorrProjection.cpp
    src/AdaptiveGating.cpp
    src/MaskGenerator.cpp
    src/SpatialReliability.cpp
    src/ChannelReliability.cpp
    src/DCFSolver.cpp
    src/UpdatedCSRTTracker.cpp
)

# Header files
set(HEADERS
    inc/Config.hpp
    inc/HOGExtractor.hpp
    inc/DeepFeatureExtractor.hpp
    inc/CorrProjection.hpp
    inc/AdaptiveGating.hpp
    inc/MaskGenerator.hpp
    inc/SpatialReliability.hpp
    inc/ChannelReliability.hpp
    inc/DCFSolver.hpp
    inc/UpdatedCSRTTracker.hpp
)

# Create library
add_library(updated_csrt_lib STATIC ${SOURCES} ${HEADERS})

# Link OpenCV
target_link_libraries(updated_csrt_lib PUBLIC ${OpenCV_LIBS})

# Compiler flags
if(MSVC)
    target_compile_options(updated_csrt_lib PRIVATE /W4 /O2)
else()
    target_compile_options(updated_csrt_lib PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Main executable
add_executable(updated_csrt_demo src/main.cpp)
target_link_libraries(updated_csrt_demo updated_csrt_lib ${OpenCV_LIBS})

# OTB-100 evaluator
add_executable(otb_eval src/otb_eval.cpp)
target_link_libraries(otb_eval updated_csrt_lib ${OpenCV_LIBS})

# Basketball quick test
add_executable(test_basketball src/test_basketball.cpp)
target_link_libraries(test_basketball updated_csrt_lib ${OpenCV_LIBS})

# Simple dual tracker (OpenCV CSRT + Deep)
# Use built-in OpenCV TrackerCSRT, no need to recompile source
add_executable(simple_dual src/simple_dual_tracker.cpp)
target_link_libraries(simple_dual ${OpenCV_LIBS})

# Installation
install(TARGETS updated_csrt_lib updated_csrt_demo
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY inc/ DESTINATION include/updated_csrt)

# Print summary
message(STATUS "================================================================================")
message(STATUS "Updated CSRT Tracker Configuration")
message(STATUS "================================================================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "================================================================================")
