cmake_minimum_required(VERSION 3.16)
project(csrt_otb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video tracking)

add_executable(csrtpure
    src/main_pcsrt.cpp)

target_link_libraries(csrtpure PRIVATE ${OpenCV_LIBS})
