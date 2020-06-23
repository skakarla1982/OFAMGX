cmake_minimum_required(VERSION 2.8.9)
project(ofamgx)

# Set compiler for mpi
set(CMAKE_CXX_COMPILER mpicxx)

# Set c++11 flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 \
                        support. Please use a different C++ compiler.")
endif()

# Link amgx library;
link_directories(lib)

# Declare a few variables
set(CMAKE_BUILD_TYPE Release)

# find CUDA library
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CUDA_LIBRARIES})

# find MPI library
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})

# Add include directories
include_directories(include)

# Add plugin directory
add_subdirectory(plugins)

# Set SRC variable
set(SRC
    amgx/AmgXSolver.cpp
)

# Generate the shared library from the sources
add_library(ofamgx SHARED ${SRC})
target_link_libraries(ofamgx ${LIBS} amgxsh mpi)
