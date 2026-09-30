cmake_minimum_required(VERSION 3.16)
project(fast_nnue_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(nnue_train src/main.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(nnue_train PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
  target_compile_options(nnue_train PRIVATE /O2 /EHsc)
else()
  target_compile_options(nnue_train PRIVATE -O3 -ffast-math -funroll-loops)
endif()
