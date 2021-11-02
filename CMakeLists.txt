cmake_minimum_required(VERSION 3.0)
project(MUI_TEST_FRAMEWORK VERSION 1.0 DESCRIPTION "Multiscale Universal Interface Testing Framework" LANGUAGES CXX)

find_package(MUI CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
find_library(LIBRT rt)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${MPI_INCLUDE_PATH})
if(MUI_BASE_DIR)
  include_directories(${MUI_INCLUDE_DIR})
endif()

add_executable(MUI_Testing MUI_testing.cpp)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(MUI_Testing PUBLIC "-pthread")
endif()

if(CMAKE_THREAD_LIBS_INIT AND LIBRT)
  target_link_libraries(MUI_Testing ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBRT} ${CMAKE_THREAD_LIBS_INIT})
elseif(CMAKE_THREAD_LIBS_INIT AND NOT LIBRT)
  target_link_libraries(MUI_Testing ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
elseif(LIBRT AND NOT CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(MUI_Testing ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBRT})
else()
  target_link_libraries(MUI_Testing ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES})
endif()
