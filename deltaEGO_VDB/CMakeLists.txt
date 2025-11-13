cmake_minimum_required(VERSION 3.15)
project(ego_vdb_py_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

set(CORE_SOURCES
    src/bindings.cpp
    VAD/VAD_customVDB.cpp
)

pybind11_add_module(core MODULE ${CORE_SOURCES})

target_include_directories(core PRIVATE
    ${pybind11_INCLUDE_DIRS}
    VAD           # VAD_customVDB.hpp
    ThirdParty    # nlohmann/json.hpp
)

install(TARGETS core
    LIBRARY DESTINATION deltaEGO_VDB
)