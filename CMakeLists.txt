cmake_minimum_required(VERSION 3.16)
project(csrt_otb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video tracking)

add_executable(eldercare_tracking_csrt_only
    src/main_csrt_only.cpp)

target_link_libraries(eldercare_tracking_csrt_only PRIVATE ${OpenCV_LIBS})
