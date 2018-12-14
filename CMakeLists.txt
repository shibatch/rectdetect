cmake_minimum_required(VERSION 3.1)
project( rect )

if (MSVC)
  # Edit the following two lines

  include_directories("D:/opencv/build/include")
  set(OpenCV_DIR "D:/opencv/mybuild")

  # Then, execute the following commands in Developer Command Prompt for VS 2017

  # mkdir build & cd build
  # cmake -G"Visual Studio 15 2017 Win64" ..
  # cmake --build . --config Release

  file(GLOB OpenCV_LIBS "${OpenCV_DIR}/lib/Release/*.lib")
  file(GLOB_RECURSE OpenCV_3RDPARTY "${OpenCV_DIR}/3rdparty/*.lib")

  foreach(item ${OpenCV_3RDPARTY})
    list(APPEND OpenCV_LIBS ${item})
  endforeach()

  foreach(msg ${OpenCV_LIBS})
    message(STATUS "OpenCV libs : " "${msg}")
  endforeach()

  include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include")
  set(LIBOPENCL "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/OpenCL.lib")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
else()
  find_package( OpenCV REQUIRED )

  find_library( LIBOPENCL OpenCL )
  include_directories( ${OPENCL_INCLUDE_DIRS} )
endif()

#

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TARGET_CLTOHEADER "cltoheader")
add_executable(${TARGET_CLTOHEADER} cltoheader.c)

#

add_custom_command(OUTPUT oclimgutil_cl.h
  COMMENT "Generating oclimgutil_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclimgutil.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclimgutil_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)

add_custom_command(OUTPUT oclpolyline_cl.h
  COMMENT "Generating oclpolyline_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclpolyline.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclpolyline_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)

add_custom_command(OUTPUT oclrect_cl.h
  COMMENT "Generating oclrect_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclrect.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclrect_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)

#

add_library(helper_obj OBJECT oclhelper.c helper.c)

add_library(oclimgutil_obj OBJECT oclimgutil.c oclimgutil.h ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)
add_dependencies(oclimgutil_obj oclimgutil_cl.h_generated)

add_library(oclpolyline_obj OBJECT oclpolyline.c oclpolyline.h ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)
add_dependencies(oclpolyline_obj oclpolyline_cl.h_generated)

add_library(oclrect_obj OBJECT oclrect.c oclrect.h ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)
add_dependencies(oclrect_obj oclrect_cl.h_generated)

#

add_executable( poly poly.cpp $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>)
target_link_libraries( poly ${OpenCV_LIBS} ${LIBOPENCL} )

add_executable( rect rect.cpp $<TARGET_OBJECTS:oclrect_obj> $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>)
target_link_libraries( rect ${OpenCV_LIBS} ${LIBOPENCL} )

add_executable( vidrect vidrect.cpp $<TARGET_OBJECTS:oclrect_obj> $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>)
target_link_libraries( vidrect ${OpenCV_LIBS} ${LIBOPENCL} )

add_executable( vidpoly vidpoly.cpp $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>)
target_link_libraries( vidpoly ${OpenCV_LIBS} ${LIBOPENCL} )

add_executable( videotest videotest.cpp)
target_link_libraries( videotest ${OpenCV_LIBS} )
