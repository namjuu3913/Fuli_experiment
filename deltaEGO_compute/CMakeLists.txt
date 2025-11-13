cmake_minimum_required(VERSION 3.15)
project(ego_vdb_py_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED) # for multithreading

set(CORE_SOURCES
    src/bindings.cpp
    compute/EGO_compute.cpp
)

pybind11_add_module(_core MODULE ${CORE_SOURCES})

target_include_directories(_core PRIVATE
    ${pybind11_INCLUDE_DIRS}
    compute      # EGO_compute.hpp, VAD.hpp
    src
    ThirdParty   
)

target_link_libraries(_core PRIVATE Threads::Threads)

install(TARGETS _core
    LIBRARY DESTINATION deltaEGO_compute
)