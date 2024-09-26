cmake_minimum_required(VERSION 3.15)
project(follower_cpp)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(follower follower.cpp config.cpp)
pybind11_add_module(config config.cpp)

target_link_libraries(follower PRIVATE onnxruntime)
set_property(TARGET follower PROPERTY CXX_STANDARD 17)