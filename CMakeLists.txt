cmake_minimum_required(VERSION 3.10)
project(TP_contours)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(tp_contours main.cpp)
target_link_libraries(tp_contours ${OpenCV_LIBS})