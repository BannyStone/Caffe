# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/SSD/zhoulei/workspace/Caffes/caffe_nd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake

# Include any dependencies generated for this target.
include tools/CMakeFiles/extract_features.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/extract_features.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/extract_features.dir/flags.make

tools/CMakeFiles/extract_features.dir/extract_features.cpp.o: tools/CMakeFiles/extract_features.dir/flags.make
tools/CMakeFiles/extract_features.dir/extract_features.cpp.o: ../tools/extract_features.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/CMakeFiles/extract_features.dir/extract_features.cpp.o"
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/extract_features.dir/extract_features.cpp.o -c /media/SSD/zhoulei/workspace/Caffes/caffe_nd/tools/extract_features.cpp

tools/CMakeFiles/extract_features.dir/extract_features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_features.dir/extract_features.cpp.i"
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/SSD/zhoulei/workspace/Caffes/caffe_nd/tools/extract_features.cpp > CMakeFiles/extract_features.dir/extract_features.cpp.i

tools/CMakeFiles/extract_features.dir/extract_features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_features.dir/extract_features.cpp.s"
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/SSD/zhoulei/workspace/Caffes/caffe_nd/tools/extract_features.cpp -o CMakeFiles/extract_features.dir/extract_features.cpp.s

tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.requires:
.PHONY : tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.requires

tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.provides: tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/extract_features.dir/build.make tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.provides.build
.PHONY : tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.provides

tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.provides.build: tools/CMakeFiles/extract_features.dir/extract_features.cpp.o

# Object files for target extract_features
extract_features_OBJECTS = \
"CMakeFiles/extract_features.dir/extract_features.cpp.o"

# External object files for target extract_features
extract_features_EXTERNAL_OBJECTS =

tools/extract_features: tools/CMakeFiles/extract_features.dir/extract_features.cpp.o
tools/extract_features: tools/CMakeFiles/extract_features.dir/build.make
tools/extract_features: lib/libcaffe.so
tools/extract_features: lib/libproto.a
tools/extract_features: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libglog.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libglog.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/extract_features: /usr/lib/libsnappy.so
tools/extract_features: /usr/local/cuda/lib64/libcudart.so
tools/extract_features: /usr/local/cuda/lib64/libcurand.so
tools/extract_features: /usr/local/cuda/lib64/libcublas.so
tools/extract_features: /usr/local/cuda/lib64/libcudnn.so
tools/extract_features: /usr/local/lib/libopencv_highgui.so.2.4.13
tools/extract_features: /usr/local/lib/libopencv_imgproc.so.2.4.13
tools/extract_features: /usr/local/lib/libopencv_core.so.2.4.13
tools/extract_features: /usr/local/cuda/lib64/libcudart.so
tools/extract_features: /usr/local/cuda/lib64/libnppc.so
tools/extract_features: /usr/local/cuda/lib64/libnppi.so
tools/extract_features: /usr/local/cuda/lib64/libnpps.so
tools/extract_features: /usr/lib/liblapack_atlas.so
tools/extract_features: /usr/lib/libcblas.so
tools/extract_features: /usr/lib/libatlas.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/extract_features: /media/SSD/zhoulei/workspace/Tools/openmpi/lib/libmpi_cxx.so
tools/extract_features: /media/SSD/zhoulei/workspace/Tools/openmpi/lib/libmpi.so
tools/extract_features: tools/CMakeFiles/extract_features.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable extract_features"
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_features.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/extract_features.dir/build: tools/extract_features
.PHONY : tools/CMakeFiles/extract_features.dir/build

tools/CMakeFiles/extract_features.dir/requires: tools/CMakeFiles/extract_features.dir/extract_features.cpp.o.requires
.PHONY : tools/CMakeFiles/extract_features.dir/requires

tools/CMakeFiles/extract_features.dir/clean:
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools && $(CMAKE_COMMAND) -P CMakeFiles/extract_features.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/extract_features.dir/clean

tools/CMakeFiles/extract_features.dir/depend:
	cd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/SSD/zhoulei/workspace/Caffes/caffe_nd /media/SSD/zhoulei/workspace/Caffes/caffe_nd/tools /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools /media/SSD/zhoulei/workspace/Caffes/caffe_nd/build_cmake/tools/CMakeFiles/extract_features.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/extract_features.dir/depend

