# Install script for directory: /home/leizhou/Caffes/caffe_nd/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/leizhou/Caffes/caffe_nd/build_cmake/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/home/leizhou/Caffes/caffe_nd/python/bn_convert_style.py"
    "/home/leizhou/Caffes/caffe_nd/python/detect.py"
    "/home/leizhou/Caffes/caffe_nd/python/gen_bn_inference.py"
    "/home/leizhou/Caffes/caffe_nd/python/classify.py"
    "/home/leizhou/Caffes/caffe_nd/python/draw_net.py"
    "/home/leizhou/Caffes/caffe_nd/python/polyak_average.py"
    "/home/leizhou/Caffes/caffe_nd/python/convert_to_fully_conv.py"
    "/home/leizhou/Caffes/caffe_nd/python/requirements.txt"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/home/leizhou/Caffes/caffe_nd/python/caffe/detector.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/net_spec_bk.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/draw.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/pycaffe.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/io.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/net_spec.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/classifier.py"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/__init__.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         RPATH "/home/leizhou/Caffes/caffe_nd/build_cmake/install/lib:/usr/lib/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/home/leizhou/Caffes/caffe_nd/build_cmake/lib/_caffe.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         OLD_RPATH "/home/leizhou/Caffes/caffe_nd/build_cmake/lib:/usr/lib/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64::::::::"
         NEW_RPATH "/home/leizhou/Caffes/caffe_nd/build_cmake/install/lib:/usr/lib/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/home/leizhou/Caffes/caffe_nd/python/caffe/imagenet"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/proto"
    "/home/leizhou/Caffes/caffe_nd/python/caffe/test"
    )
endif()

