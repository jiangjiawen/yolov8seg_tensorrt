cmake_minimum_required(VERSION 2.9)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(yolov8SegSample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")


# cuda
set(cuda_inc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
set(cuda_lib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/x64")
include_directories(${cuda_inc})
link_directories(${cuda_lib})

# tensorrt
set(tensorrt_inc "C:/TensorRT-8.5.1.7/include")
set(tensorrt_lib "C:/TensorRT-8.5.1.7/lib")
include_directories(${tensorrt_inc})
link_directories(${tensorrt_lib})

# opencv
include_directories("C:/cvtools/opencv48/opencv/build/include")
link_directories("C:/cvtools/opencv48/opencv/build/x64/vc16/lib")

# include utils
include_directories("${CMAKE_SOURCE_DIR}/cuda_utils")
file(GLOB UTILS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils/*.cpp)

add_executable(yolov8SegSample main.cpp yolov8_seg.hpp ${UTILS_FILES})
target_link_libraries(yolov8SegSample nvinfer cudart opencv_world480)