cmake_minimum_required(VERSION 3.8)
project(image_classifier)
set(CMAKE_CXX_STANDARD 11)


file(GLOB_RECURSE SOURCES src/*.cpp lib/*.cpp )
file(GLOB_RECURSE HEADERS include/*.h lib/*.h)

find_package(OpenCV 3.0)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
endif ()

#link_directories("/home/jintian/caffe/build/install/include/caffe")
#find_library(CAFFE_LIBRARY
#        NAMES libcaffe caffe
#        HINTS "/home/jintian/caffe/build/install/lib")
#if (caffe_FOUND)
#    message(STATUS "we just found caffe!!")
#endif ()


# ------------------ find Caffe and include directories ------------------------
set(Caffe_INCLUDE_DIRS "/home/jintian/caffe/build/install/include")
set(Caffe_LIBRARIES "/home/jintian/caffe/build/install/lib/libcaffe.so")
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
if (Caffe_FOUND)
    message(STATUS "[caffe] we just found caffe.")
    message(STATUS ${Caffe_INCLUDE_DIRS})
endif ()
# ------------------ we must mannully specific directories of caffe -----------------


# ----------------- find CUDA to compatible caffe build with CUDA ------------
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "[cuda] we found CUDA.")
    message(STATUS ${CUDA_INCLUDE_DIRS})
endif ()
include_directories(${CUDA_INCLUDE_DIRS})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/lib)

set(BUILD_SHARED_LIBS OFF)

set(SOURCE_FILES src/main.cpp src/classifier.cpp src/classifier.h)

add_executable(image_classifier ${SOURCE_FILES} ${SOURCES} ${HEADERS})
target_link_libraries(image_classifier ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${CUDA_LIBS})