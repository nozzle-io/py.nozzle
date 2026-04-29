cmake_minimum_required(VERSION 3.20)

if(MSVC)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

project(nozzle_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "" FORCE)
    endif()
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Development.Module
                                     OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# nozzle needs plog but may not find it if not built from its own tree.
# pass NOZZLE_BUILD_EXAMPLES=OFF and NOZZLE_BUILD_TESTS=OFF to skip those.
set(NOZZLE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NOZZLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NOZZLE_BUILD_OPENGL OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()
add_subdirectory(libs/nozzle)

nanobind_add_module(_nozzle STABLE_ABI src/nozzle.cpp)
target_link_libraries(_nozzle PRIVATE nozzle)
target_include_directories(_nozzle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/nozzle/include)

install(TARGETS _nozzle LIBRARY DESTINATION nozzle)
