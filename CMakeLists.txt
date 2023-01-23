cmake_minimum_required(VERSION 3.10)
project(BerlinTest)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(TensorRT_DIR /usr/lib/aarch64-linux-gnu/)

include_directories(/usr/include/opencv4/)
include_directories(/usr/lib/aarch64-linux-gnu)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(BerlinTest main.cpp)

target_include_directories(BerlinTest PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_DIR})
target_link_libraries(BerlinTest ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_link_libraries(BerlinTest nvinfer nvonnxparser)