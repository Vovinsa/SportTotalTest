cmake_minimum_required(VERSION 3.10)
project(SportTotalTest)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

include_directories(/usr/include/opencv4)
include_directories(/usr/lib/aarch64-linux-gnu)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(SportTotalTest main.cpp infer/logger.cpp infer/TRTInference.cpp)

target_include_directories(SportTotalTest PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(SportTotalTest ${OpenCV_LIBS} ${CUDA_LIBRARIES})
target_link_libraries(SportTotalTest nvinfer nvonnxparser microhttpd)