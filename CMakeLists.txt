cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER /opt/cuda/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/cuda/bin/g++)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/

execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_INCLUDE_LIBS)

execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)


execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TF_CFLAGS)

execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TF_LFLAGS)


message("AAAAAAAAAAAAAAAAAAABBBB")
message("Tensorflow_INCLUDE_DIRS " ${Tensorflow_INCLUDE_DIRS})
message("Tensorflow_INCLUDE_LIBS " ${Tensorflow_INCLUDE_LIBS})

message("TF_CFLAGS "${TF_CFLAGS})
message("TF_LFLAGS "${TF_LFLAGS})


find_package(CUDA)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${TF_CFLAGS} ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${TF_LFLAGS} ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-I/usr/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-L/usr/lib/python3.7/site-packages/tensorflow/ -ltensorflow_framework ${CMAKE_CXX_FLAGS}")





# if GCC > 5
#if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
#  message("C++ compiler above version 5")
#  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
#endif()

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})

#create library
cuda_add_library(
    dense SHARED
    cuda_op_kernel.cu
    dense.cc)
    
    #OPTIONS -gencode=arch=compute_20,code=sm_20)

#add_library(dense SHARED dense.cc)
