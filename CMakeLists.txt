cmake_minimum_required(VERSION 3.18)
project(GPU_Image_Filter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(GPU_Image_Filter main.cpp image_kernels.cu colormap.hpp)
target_include_directories(GPU_Image_Filter PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GPU_Image_Filter PRIVATE ${OpenCV_LIBS})
set_target_properties(GPU_Image_Filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
